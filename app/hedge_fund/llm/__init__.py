"""LLM submodule."""

from app.hedge_fund.llm.gateway import (
    INVESTMENT_SIGNAL_SCHEMA,
    LLMGatewayProtocol,
    OpenAIGateway,
    get_gateway,
    get_investment_analysis,
    set_gateway,
)

__all__ = [
    "LLMGatewayProtocol",
    "OpenAIGateway",
    "get_gateway",
    "set_gateway",
    "get_investment_analysis",
    "INVESTMENT_SIGNAL_SCHEMA",
]
