"""
OpenAI Batch API for bulk processing.

Provides batch submission, status checking, and result collection
with 50% cost savings compared to real-time API calls.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.services.openai.client import get_client
from app.services.openai.config import (
    TaskType,
    get_settings,
    get_task_config,
    is_reasoning_model,
)
from app.services.openai.generate import (
    build_prompt,
    calculate_safe_output_tokens,
)
from app.services.openai.prompts import get_instructions
from app.services.openai.schemas import TASK_SCHEMAS

logger = get_logger("openai.batch")


async def submit_batch(
    task: TaskType | str,
    items: list[dict[str, Any]],
    *,
    model: str | None = None,
) -> str | None:
    """
    Submit a batch job for bulk processing.
    
    Each item in the list represents a separate API call to be made.
    Batch API provides ~50% cost savings and 24-hour completion window.
    
    Args:
        task: Type of task to run
        items: List of context dicts, each with at least 'symbol'
        model: Override default model
    
    Returns:
        Batch job ID, or None on failure
    
    Example:
        >>> batch_id = await submit_batch(
        ...     task=TaskType.BIO,
        ...     items=[
        ...         {"symbol": "AAPL", "name": "Apple Inc."},
        ...         {"symbol": "MSFT", "name": "Microsoft Corp."},
        ...     ]
        ... )
    """
    client = await get_client()
    if not client or not items:
        return None
    
    # Normalize task
    if isinstance(task, str):
        task = TaskType(task)
    
    settings = get_settings()
    model = model or settings.default_model
    instructions = get_instructions(task)
    config = get_task_config(task)
    
    # Generate unique batch run ID
    batch_run_id = uuid.uuid4().hex[:8]
    
    # Build JSONL lines
    jsonl_lines: list[str] = []
    
    for i, item in enumerate(items):
        # Build custom_id
        if task == TaskType.PORTFOLIO and item.get("custom_id"):
            custom_id = item["custom_id"]
        else:
            symbol = item.get("symbol", "unknown")
            agent_id = item.get("agent_id", "")
            if agent_id:
                # Agent batch: "batch_run_id:symbol:agent_id:task"
                custom_id = f"{batch_run_id}:{symbol}:{agent_id}:{task.value}"
            else:
                # Standard: "batch_run_id:symbol:idx:task"
                custom_id = f"{batch_run_id}:{symbol}:{i}:{task.value}"
        
        # Build prompt
        prompt = build_prompt(task, item)
        
        # Calculate safe output tokens
        safe_output, overflow = calculate_safe_output_tokens(
            model=model,
            instructions=instructions,
            prompt=prompt,
            desired_output=config.default_max_tokens,
            task=task,
        )
        
        if overflow:
            logger.warning(f"Skipping {item.get('symbol', 'unknown')} in batch: input too large")
            continue
        
        # Build request body
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": safe_output,
            "store": False,
        }
        
        # Use low effort for reasoning models in batch
        if is_reasoning_model(model):
            body["reasoning"] = {"effort": "low"}
        
        # All tasks use structured outputs
        body["text"] = {"format": TASK_SCHEMAS[task]}
        
        jsonl_lines.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }))
    
    if not jsonl_lines:
        logger.warning("No valid items for batch after filtering")
        return None
    
    try:
        # Write JSONL to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(jsonl_lines))
            temp_path = f.name
        
        # Upload batch file
        with open(temp_path, "rb") as f:
            batch_file = await client.files.create(file=f, purpose="batch")
        
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)
        
        # Create batch job
        batch = await client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "task": task.value,
                "model": model,
                "count": str(len(jsonl_lines)),
                "batch_run_id": batch_run_id,
            },
        )
        
        logger.info(f"Created batch {batch.id}: {len(jsonl_lines)} {task.value} items")
        return batch.id
    
    except Exception as e:
        logger.error(f"Failed to create batch: {e}")
        return None


async def check_batch(batch_id: str) -> dict[str, Any] | None:
    """
    Check status of a batch job.
    
    Returns:
        Dict with batch status and counts, or None on error
    """
    client = await get_client()
    if not client:
        return None
    
    try:
        batch = await client.batches.retrieve(batch_id)
        
        completed = batch.request_counts.completed if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        
        # Log status for worker visibility
        logger.info(
            f"[BATCH] {batch_id[:16]}... status={batch.status}, "
            f"{completed}/{total} completed, {failed} failed"
        )
        
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "completed_count": completed,
            "failed_count": failed,
            "total_count": total,
            "counts": {
                "total": total,
                "completed": completed,
                "failed": failed,
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
    except Exception as e:
        logger.error(f"Failed to check batch {batch_id}: {e}")
        return None


async def collect_batch(batch_id: str) -> list[dict[str, Any]] | None:
    """
    Collect results from a completed batch job.
    
    Parses the output file and returns a list of result dicts with:
    - custom_id: The original request ID
    - symbol: Extracted from custom_id
    - result: Parsed JSON response
    - failed: True if request failed
    - error: Error message if failed
    
    Returns:
        List of result dicts, or None on error
    """
    client = await get_client()
    if not client:
        return None
    
    try:
        batch = await client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            logger.warning(f"Batch {batch_id} not completed: {batch.status}")
            return None
        
        if not batch.output_file_id:
            logger.warning(f"Batch {batch_id} has no output file")
            return None
        
        # Download output file
        content = await client.files.content(batch.output_file_id)
        lines = content.text.strip().split("\n")
        
        results: list[dict[str, Any]] = []
        
        for line in lines:
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
                custom_id = item.get("custom_id", "")
                
                # Parse custom_id to extract symbol
                # Format: "batch_run_id:symbol:agent_id:task" or "batch_run_id:symbol:idx:task"
                parts = custom_id.split(":")
                symbol = parts[1] if len(parts) >= 2 else ""
                
                # Check for errors
                error = item.get("error")
                if error:
                    results.append({
                        "custom_id": custom_id,
                        "symbol": symbol,
                        "result": None,
                        "failed": True,
                        "error": error.get("message", str(error)),
                    })
                    continue
                
                # Extract response
                response = item.get("response", {})
                body = response.get("body", {})
                
                # Get output text from response
                output_text = ""
                if "output" in body:
                    output = body["output"]
                    if isinstance(output, list):
                        for block in output:
                            if block.get("type") == "message":
                                for content in block.get("content", []):
                                    if content.get("type") == "output_text":
                                        output_text = content.get("text", "")
                                        break
                    elif isinstance(output, str):
                        output_text = output
                
                # Also check output_text directly
                if not output_text:
                    output_text = body.get("output_text", "")
                
                # Parse JSON result
                if output_text:
                    try:
                        parsed = json.loads(output_text)
                        results.append({
                            "custom_id": custom_id,
                            "symbol": symbol,
                            "result": parsed,
                            "failed": False,
                            "error": None,
                        })
                    except json.JSONDecodeError as e:
                        results.append({
                            "custom_id": custom_id,
                            "symbol": symbol,
                            "result": output_text,
                            "failed": True,
                            "error": f"JSON parse error: {e}",
                        })
                else:
                    results.append({
                        "custom_id": custom_id,
                        "symbol": symbol,
                        "result": None,
                        "failed": True,
                        "error": "Empty output",
                    })
            
            except Exception as e:
                logger.warning(f"Failed to parse batch result line: {e}")
                continue
        
        # Log detailed results for worker visibility
        success_count = sum(1 for r in results if not r.get("failed"))
        failed_count = sum(1 for r in results if r.get("failed"))
        logger.info(
            f"[BATCH] Collected {batch_id[:16]}... - "
            f"{success_count} succeeded, {failed_count} failed"
        )
        return results
    
    except Exception as e:
        logger.error(f"Failed to collect batch {batch_id}: {e}")
        return None


async def cancel_batch(batch_id: str) -> bool:
    """
    Cancel a batch job that hasn't completed yet.
    
    Returns:
        True if cancelled successfully, False otherwise
    """
    client = await get_client()
    if not client:
        return False
    
    try:
        await client.batches.cancel(batch_id)
        logger.info(f"Cancelled batch {batch_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel batch {batch_id}: {e}")
        return False


async def retry_failed_items(
    batch_id: str,
    task: TaskType | str,
    *,
    model: str | None = None,
) -> str | None:
    """
    Retry failed items from a batch job by submitting a new batch.
    
    Args:
        batch_id: Original batch ID
        task: Task type for the items
        model: Override model
    
    Returns:
        New batch ID for retry, or None if no failures/error
    """
    # Collect original results
    results = await collect_batch(batch_id)
    if not results:
        return None
    
    # Filter failed items
    failed_items: list[dict[str, Any]] = []
    for result in results:
        if result.get("failed"):
            # Reconstruct context from custom_id
            custom_id = result.get("custom_id", "")
            parts = custom_id.split(":")
            if len(parts) >= 2:
                symbol = parts[1]
                agent_id = parts[2] if len(parts) >= 3 and not parts[2].isdigit() else None
                
                item: dict[str, Any] = {"symbol": symbol}
                if agent_id:
                    item["agent_id"] = agent_id
                failed_items.append(item)
    
    if not failed_items:
        logger.info(f"No failed items to retry in batch {batch_id}")
        return None
    
    logger.info(f"Retrying {len(failed_items)} failed items from batch {batch_id}")
    
    return await submit_batch(
        task=task,
        items=failed_items,
        model=model,
    )
