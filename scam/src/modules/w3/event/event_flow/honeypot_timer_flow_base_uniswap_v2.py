from .event_flow import *
from ..honeypot_event import *
from ..honeypot_event import HoneypotEvent
from ...exchange.uniswap_v2_base import UniswapV2Base
import random

class HoneypotTimerFlowBaseUniswapV2(EventFlow):
    def __init__(self, w3, scanner, exchange: UniswapV2Base, account):
        self.w3 = w3
        self.scanner = scanner
        self.exchange = exchange
        self.account = account
        self.directory = 'honeypot_timer_flow'
        self.setup_logs()
        self.weth_address = w3.to_checksum_address("0x4200000000000000000000000000000000000006")
        self.usdc_address = w3.to_checksum_address("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
        self.weth = Token(self.weth_address, w3, scanner)
        self.usdc = Token(self.usdc_address, w3, scanner)
        self.weth_usdc_pair = Pair(self.weth, self.usdc, w3, scanner, exchange)
        self.SLIPPAGE_VALUES = [3, 5]
        self.WAIT_TIMES_MINUTES = list(range(5, 10))
        self.BUY_AMOUNTS = [0.0002]
        self.buy_amount = Decimal(random.choice(self.BUY_AMOUNTS))
        self.wait_time_minutes = random.choice(self.WAIT_TIMES_MINUTES)
        self.wait_time_seconds = self.wait_time_minutes * 60
        
    def handle_event(self, event_data):
        # distinguish between tokens
        token, return_flag = self.find_weth_token(event_data)
        if return_flag:
            return

        # Get the logger for this token
        self.get_token_logger(token, directory='honeypot_timer_flow')
        
        # Create the event object
        event, return_flag = self.create_event(token, event_class=HoneypotEvent)
        if return_flag:
            return
        
        # create pair object 
        pair, return_flag = self.create_pair(token, event)
        if return_flag:
            self.cleanup_logs(token.address)
            return
        
        # check pair liquidity
        return_flag = self.pair_liquidity_check(event)
        if return_flag:
            return
        
        # perform security checks
        return_flag = self.security_checks(event)
        if return_flag:
            return
               
        return_flag = self.llm_check(event)
        if return_flag:
            return
        
        # Do a code similarity check
        similar_flag, return_flag = self.code_similarity_check(token.code)
        if return_flag:
            return
        if similar_flag:
            # update buy amount
            self.buy_amount = Decimal('0.001')
            # log
            event.logger.info("Code similarity detected, updating buy amount to 0.001")
        
        # Transact
        try:
            outcome = self.transact(event)
            self.logger.info(f"Transaction outcome: {outcome}")
        except Exception as e:
            self.logger.error(f"Error during transaction process: {str(e)}")
            return
        
        # Get an account observation after the transaction
        try:
            event.account_value_post_transaction = self.get_total_account_value_eth()
            event.post_transaction_observation_timestamp = int(time.time())
            event.logger.info(f"Account value observation: {event.account_value_post_transaction} ETH")
            event.logger.info(f"Observation timestamp: {event.post_transaction_observation_timestamp}")
        except Exception as e:
            self.logger.error(f"Error getting account value post-transaction: {str(e)}")

        # Handle transaction outcome
        if outcome == "Buy failed":
            self.logger.warning("Buy transaction failed.")
            self.cleanup_logs(token.address)
            return
        elif outcome == "Transaction complete" and event.short_term_outcome == "Successful Sell" and not similar_flag:
            self.cleanup_logs(token.address)
        else:
            self.logger.info(f"Transaction outcome: {outcome}")

        # Save the event object as a JSON file
        try:
            data_path = f"data/honeypot_timer_flow/{token.address}.json"
            with open(data_path, "w") as event_file:
                json.dump(event.to_dict(), event_file, default=str, indent=4)
            self.logger.info(f"Event data saved to {data_path}")
        except Exception as e:
            self.logger.error(f"Error saving event data to JSON: {str(e)}")

    def transact(self, event: HoneypotEvent):
        # Initialize total gas costs
        total_buy_gas_cost_eth = Decimal('0')
        total_sell_gas_cost_eth = Decimal('0')

        # Get an account observation before the transaction
        event.account_value_pre_transaction = self.get_total_account_value_eth()
        event.pre_transaction_observation_timestamp = int(time.time())
        # log
        event.logger.info(f"Account value observation: {event.account_value_pre_transaction} ETH")
        event.logger.info(f"Observation timestamp: {event.pre_transaction_observation_timestamp}")

        # Buy attempt
        event.logger.info("Initiating buy procedure.")
        for slippage in self.SLIPPAGE_VALUES:
            event.logger.info(f"Trying buy with {slippage}% slippage...")
            slippage_decimal = Decimal(slippage) / Decimal('100')
            try:
                swap_result = self.exchange.swap_tokens(
                    w3=self.w3,
                    account=self.account,
                    from_token=self.weth,
                    to_token=event.token,
                    event=event,
                    amount_in_tokens=self.buy_amount,
                    slippage_tolerance=slippage_decimal,
                    gas_speed="medium"  # or "low"/"high" depending on your preference
                )
                # Populate event details
                if swap_result["swap_status"] == 1:
                    total_buy_gas_cost_eth += Decimal(swap_result.get("total_gas_cost_eth", 0))
                    event.buy_gas_used += (swap_result.get("swap_gas_cost_eth", 0) + swap_result.get("approval_gas_cost_eth", 0)) * 10**18  # in Wei
                    event.buy_slippage = float(slippage_decimal)
                    event.successful_buy_hashes.append(swap_result["swap_tx_hash"])
                    event.logger.info("Buy successful.")
                    event.amount_in = float(self.buy_amount)
                    event.amount_in_raw = to_base_units(self.buy_amount, self.weth.decimals)
                    break
                else:
                    event.amount_in = float(self.buy_amount)
                    event.failed_buy_hashes.append(swap_result["swap_tx_hash"])
                    total_buy_gas_cost_eth += Decimal(swap_result.get("total_gas_cost_eth", 0))
                    event.buy_gas_used += (swap_result.get("swap_gas_cost_eth", 0) + swap_result.get("approval_gas_cost_eth", 0)) * 10**18  # in Wei
                    event.logger.error("Buy failed.")
            except Exception as e:
                event.logger.error(f"Exception during buy attempt: {str(e)}")

        if len(event.successful_buy_hashes) == 0:
            event.logger.error("Failed to buy at all slippage values.")
            return "Buy failed"

        # Wait for some time
        event.logger.info(f"Waiting for {self.wait_time_minutes} minutes...")
        time.sleep(self.wait_time_seconds)

        # Sell attempt
        event.logger.info("Initiating sell procedure.")
        for slippage in self.SLIPPAGE_VALUES:
            event.logger.info(f"Trying sell with {slippage}% slippage...")
            slippage_decimal = Decimal(slippage) / Decimal('100')
            try:
                swap_result = self.exchange.swap_tokens(
                    w3=self.w3,
                    account=self.account,
                    from_token=event.token,
                    to_token=self.weth,
                    event=event,
                    amount_in_tokens=None,  # Sell all
                    slippage_tolerance=slippage_decimal,
                    gas_speed="medium"  # or "low"/"high"
                )
                # Populate event details
                if swap_result["swap_status"] == 1:
                    total_sell_gas_cost_eth += Decimal(swap_result.get("total_gas_cost_eth", 0))
                    event.sell_gas_used += (swap_result.get("swap_gas_cost_eth", 0) + swap_result.get("approval_gas_cost_eth", 0)) * 10**18  # in Wei
                    event.sell_slippage = float(slippage_decimal)
                    event.amount_out = swap_result["amount_out"]
                    event.successful_sell_hashes.append(swap_result["swap_tx_hash"])
                    event.logger.info("Sell successful.")
                    break
                else:
                    event.failed_sell_hashes.append(swap_result["swap_tx_hash"])
                    total_sell_gas_cost_eth += Decimal(swap_result.get("total_gas_cost_eth", 0))
                    event.sell_gas_used += (swap_result.get("swap_gas_cost_eth", 0) + swap_result.get("approval_gas_cost_eth", 0)) * 10**18  # in Wei
                    event.amount_out = 0
                    event.logger.error("Sell failed.")
            except Exception as e:
                event.logger.error(f"Exception during sell attempt: {str(e)}")
                # if replacement transaction underpriced go with higher gas to sell
                if "underpriced" in str(e):
                    event.logger.error("Underpriced transaction, retrying with higher gas")
                    try:
                        swap_result = self.exchange.swap_tokens(
                            w3=self.w3,
                            account=self.account,
                            from_token=event.token,
                            to_token=self.weth,
                            event=event,
                            amount_in_tokens=None,  # Sell all
                            slippage_tolerance=slippage_decimal,
                            gas_limit_approve= 500000,
                            gas_limit_swap= 500000,
                            gas_speed="high"  # or "low"/"medium"
                        )
                        # Populate event details
                        if swap_result["swap_status"] == 1:
                            total_sell_gas_cost_eth += Decimal(swap_result.get("total_gas_cost_eth", 0))
                            event.sell_gas_used += (swap_result.get("swap_gas_cost_eth", 0) + swap_result.get("approval_gas_cost_eth", 0)) * 10**18  # in Wei
                            event.sell_slippage = float(slippage_decimal)
                            event.amount_out = swap_result["amount_out"]
                            event.successful_sell_hashes.append(swap_result["swap_tx_hash"])
                            event.logger.info("Sell successful.")
                            break
                        else:
                            event.failed_sell_hashes.append(swap_result["swap_tx_hash"])
                            total_sell_gas_cost_eth += Decimal(swap_result.get("total_gas_cost_eth", 0))
                            event.sell_gas_used += (swap_result.get("swap_gas_cost_eth", 0) + swap_result.get("approval_gas_cost_eth", 0)) * 10**18  # in Wei
                            event.amount_out = 0
                            event.logger.error("Sell failed.")
                    except Exception as e:
                        event.logger.error(f"Exception during sell attempt: {str(e)}")
                    

        # Calculate Gas Costs in ETH
        total_gas_cost_eth = total_buy_gas_cost_eth + total_sell_gas_cost_eth

        # Convert buy_amount and amount_out to Decimal for calculation
        buy_amount_eth = Decimal(event.amount_in)
        amount_out_eth = Decimal(event.amount_out)

        # Calculate Profit
        profit = amount_out_eth - buy_amount_eth - total_gas_cost_eth
        event.profit = float(profit)
        event.logger.info(f"Profit: {profit} ETH")

        # Calculate Yield Percentage
        if buy_amount_eth != 0:
            yield_percentage = (profit / buy_amount_eth) * Decimal('100')
        else:
            yield_percentage = Decimal('0')
        event.yield_percent = float(yield_percentage)
        event.logger.info(f"Yield percentage: {yield_percentage}%")

        # Log outcome
        if event.amount_out == 0:
            event.short_term_outcome = "Failed Sell"
            event.can_sell = False

            # Investigate failure reasons
            token_balance = event.token.get_balance(self.account.address)
            if token_balance == 0:
                event.fail_reason = "No tokens received"
            else:
                liquidity_report = self.liquidity_check_usd(self.pair, self.w3, self.scanner)

                if liquidity_report < 1:
                    event.fail_reason = "No liquidity"
                else:
                    event.fail_reason = "Unknown error"

            event.logger.warning(f"Short-term outcome: {event.short_term_outcome}")
            event.logger.warning(f"Failure reason: {event.fail_reason}")
        else:
            event.short_term_outcome = "Successful Sell"
            event.can_sell = True
            event.logger.info(f"Short-term outcome: {event.short_term_outcome}")

        return "Transaction complete"