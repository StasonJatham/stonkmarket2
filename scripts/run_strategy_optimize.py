#!/usr/bin/env python3
"""Run strategy optimization job manually."""
import asyncio
import sys
sys.path.insert(0, "/Users/karl/Code/stonkmarket")

from app.jobs.definitions import strategy_nightly_job


async def main():
    result = await strategy_nightly_job()
    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
