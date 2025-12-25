"""Agents submodule."""

from app.hedge_fund.agents.base import (
    AgentBase,
    AgentProtocol,
    CalculationAgentBase,
    LLMAgentBase,
)
from app.hedge_fund.agents.fundamentals import (
    FundamentalsAgent,
    get_fundamentals_agent,
)
from app.hedge_fund.agents.investor_persona import (
    InvestorPersonaAgent,
    get_all_persona_agents,
    get_all_personas,
    get_persona,
    get_persona_agent,
    PERSONAS,
)
from app.hedge_fund.agents.portfolio_manager import (
    PortfolioManager,
    calculate_confidence,
    calculate_consensus,
    get_portfolio_manager,
)
from app.hedge_fund.agents.risk import (
    RiskAgent,
    get_risk_agent,
)
from app.hedge_fund.agents.sentiment import (
    SentimentAgent,
    get_sentiment_agent,
)
from app.hedge_fund.agents.technicals import (
    TechnicalsAgent,
    get_technicals_agent,
)
from app.hedge_fund.agents.valuation import (
    ValuationAgent,
    get_valuation_agent,
)

__all__ = [
    # Base classes
    "AgentBase",
    "AgentProtocol",
    "CalculationAgentBase",
    "LLMAgentBase",
    # Calculation agents
    "FundamentalsAgent",
    "get_fundamentals_agent",
    "TechnicalsAgent",
    "get_technicals_agent",
    "ValuationAgent",
    "get_valuation_agent",
    "SentimentAgent",
    "get_sentiment_agent",
    "RiskAgent",
    "get_risk_agent",
    # Persona agents
    "InvestorPersonaAgent",
    "get_persona",
    "get_persona_agent",
    "get_all_personas",
    "get_all_persona_agents",
    "PERSONAS",
    # Portfolio manager
    "PortfolioManager",
    "get_portfolio_manager",
    "calculate_consensus",
    "calculate_confidence",
]
