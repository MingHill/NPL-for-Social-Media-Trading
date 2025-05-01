import logging

class Checks:
    def __init__(self, event):
        raise NotImplementedError
    
    def check(self):
        raise NotImplementedError

class SecurityFunctionPresence(Checks):
    def __init__(self, event):
        self.event = event
        # Get contract code
        self.contract_code = event.token.code
        self.functions = event.token.functions
        self.bad_functions = ["permitAllance", "_burn", "isOwner", "mod", "getHolders", "_math", "tokenSymbol", "tokenDecimals", "swap", "executeSwap", "ERC20Coefficient", "addToBlacklist", "enableTrading", "removeLimits"]
        self.warning_functions = ["claimGas"]
        self.function_combos = [("claim", "multicall", "execute")]
    def check(self):
        for warning_function in self.warning_functions:
            if warning_function in self.functions:
                self.event.logger.warning(
                    f"{warning_function} found in contract ({self.event.token.address})"
                )
                self.event.bad_functions.append(warning_function)

        for bad_function in self.bad_functions:
            if bad_function in self.functions:
                self.event.logger.warning(
                    f"{bad_function} found in contract ({self.event.token.address})"
                )
                self.event.bad_functions.append(bad_function)
                return True
            
        for combo in self.function_combos:
            if all([f in self.functions for f in combo]):
                self.event.logger.error(
                    f"Function combination {combo} found in contract ({self.event.token.address})"
                )
                self.event.bad_functions += combo
                return True
        return False
    
# Creator black list 0x1a2748128c1044963be51850afbe5ef22520da11
class SecurityBadCreator(Checks):
    def __init__(self, event):
        self.event = event
        self.creator = event.token.contract_creator
        self.blacklist = ["0x1a2748128c1044963be51850afbe5ef22520da11", "0x75F0849aF9617b8859d1E3c2bA04856298D2556D"]

    def check(self):
        if self.creator in self.blacklist:
            self.event.logger.error(
                f"Creator {self.creator} found in contract ({self.event.token.address})"
            )
            self.event.bad_lines.append(self.creator)
            return True
        return False   
    
class SecurityBadLines(Checks):
    def __init__(self, event):
        self.event = event
        # Get contract code
        self.contract_code = event.token.code
        self.warning_lines = ["library Address"]
        self.bad_lines = ["_p76234"]

    def check(self):
        for line in self.warning_lines:
            if line in self.contract_code:
                self.event.logger.warning(
                    f"{line} found in contract ({self.event.token.address})"
                )
                self.event.bad_lines.append(line)
        
        for line in self.bad_lines:
            if line in self.contract_code:
                self.event.logger.error(
                    f"{line} found in contract ({self.event.token.address})"
                )
                self.event.bad_lines.append(line)
                return True
        return False
