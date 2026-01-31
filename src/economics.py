class EvoCompEconomics:
    def estimate_total_cost(self) -> float:
        """
        Estimates the total cost of the evolutionary composition process.
        """
        # Values based on Addendum Table 2
        discovery_cost = 3024
        ai_scientist_api = 500
        automated_verification = 50

        return discovery_cost + ai_scientist_api + automated_verification
