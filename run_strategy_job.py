#!/usr/bin/env python
"""Run the strategy_nightly job."""
import asyncio
from app.jobs.quant import strategy_nightly_job

if __name__ == "__main__":
    result = asyncio.run(strategy_nightly_job())
    print(result)
