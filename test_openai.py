#!/usr/bin/env python3
"""Test OpenAI client with real API calls."""
import asyncio
from dotenv import load_dotenv
load_dotenv()

from app.services.openai import generate, submit_batch, check_batch, collect_batch, TaskType

async def test_realtime():
    print('='*60)
    print('TESTING OPENAI CLIENT - REAL-TIME API')
    print('='*60)
    
    # 1. BIO
    print('\n1. Testing BIO generation for AAPL...')
    bio = await generate(
        task=TaskType.BIO,
        context={
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'summary': 'Consumer electronics, software, and services.',
            'dip_pct': 12.3
        }
    )
    print(f'   Result: {bio}')
    print(f'   Length: {len(bio)} chars')
    assert bio and 150 <= len(bio) <= 220, f'BIO length out of range: {len(bio)}'
    print('   ✅ BIO passed')
    
    # 2. RATING 
    print('\n2. Testing RATING generation for TSLA...')
    rating = await generate(
        task=TaskType.RATING,
        context={
            'symbol': 'TSLA',
            'name': 'Tesla Inc.',
            'sector': 'Consumer Cyclical',
            'summary': 'Electric vehicles and clean energy.',
            'current_price': 180.50,
            'ref_high': 213.75,
            'dip_pct': 15.5,
            'days_below': 25,
            'pe_ratio': 85.2
        },
        json_output=True
    )
    print(f'   Result: {rating}')
    assert rating and 'rating' in rating and 'reasoning' in rating and 'confidence' in rating
    print(f'   Reasoning length: {len(rating["reasoning"])} chars')
    print('   ✅ RATING passed')
    
    # 3. SUMMARY
    print('\n3. Testing SUMMARY generation for NVDA...')
    summary = await generate(
        task=TaskType.SUMMARY,
        context={
            'symbol': 'NVDA',
            'name': 'NVIDIA Corporation',
            'description': '''NVIDIA Corporation provides graphics, and compute and networking solutions in the United States, Taiwan, China, Hong Kong, and internationally. The Graphics segment offers GeForce GPUs for gaming and PCs, the GeForce NOW game streaming service and related infrastructure, and solutions for gaming platforms; Quadro/NVIDIA RTX GPUs for enterprise workstation graphics; virtual GPU or vGPU software for cloud-based visual and virtual computing; automotive platforms for infotainment systems; and Omniverse software for building and operating metaverse and 3D internet applications.'''
        }
    )
    print(f'   Result: {summary}')
    print(f'   Length: {len(summary)} chars')
    assert summary and 300 <= len(summary) <= 550, f'SUMMARY length out of range: {len(summary)}'
    print('   ✅ SUMMARY passed')
    
    print('\n' + '='*60)
    print('ALL REAL-TIME TESTS PASSED!')
    print('='*60)


async def test_batch():
    print('\n' + '='*60)
    print('TESTING OPENAI CLIENT - BATCH API')
    print('='*60)
    
    items = [
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'dip_pct': 12.0, 'days_below': 10, 'pe_ratio': 28.5},
        {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'dip_pct': 8.5, 'days_below': 5, 'pe_ratio': 32.0},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'dip_pct': 22.0, 'days_below': 45, 'pe_ratio': 24.0},
    ]
    
    print(f'\n1. Submitting batch with {len(items)} RATING items...')
    batch_id = await submit_batch(task=TaskType.RATING, items=items)
    print(f'   Batch ID: {batch_id}')
    assert batch_id, 'Failed to submit batch'
    print('   ✅ Batch submitted')
    
    print('\n2. Polling batch status...')
    max_wait = 180  # 3 minutes max
    poll_interval = 10
    elapsed = 0
    
    while elapsed < max_wait:
        status = await check_batch(batch_id)
        print(f'   [{elapsed}s] Status: {status["status"]}, Progress: {status["completed_count"]}/{status["total_count"]}')
        
        if status['status'] == 'completed':
            print('   ✅ Batch completed')
            break
        elif status['status'] in ('failed', 'expired', 'cancelled'):
            print(f'   ❌ Batch failed: {status["status"]}')
            return
        
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    else:
        print(f'   ⏱️ Timeout after {max_wait}s, batch still processing')
        return
    
    print('\n3. Collecting batch results...')
    results = await collect_batch(batch_id)
    print(f'   Got {len(results)} results:')
    
    for r in results:
        symbol = r['symbol']
        if r['failed']:
            print(f'   ❌ {symbol}: FAILED - {r["error"]}')
        else:
            result = r['result']
            print(f'   ✅ {symbol}: {result["rating"]} (confidence: {result["confidence"]})')
            print(f'      Reasoning: {result["reasoning"][:80]}...')
    
    success_count = sum(1 for r in results if not r['failed'])
    print(f'\n   Summary: {success_count}/{len(results)} succeeded')
    print('   ✅ Batch collection complete')
    
    print('\n' + '='*60)
    print('BATCH TEST COMPLETE!')
    print('='*60)


async def main():
    await test_realtime()
    await test_batch()

if __name__ == '__main__':
    asyncio.run(main())
