import json
import os
import socket
import threading
import pickle
import logging
from pathlib import Path
from account import Account

# Create logs directory if it doesn't exist
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Define log file path
log_file = logs_dir / 'account_service.log'

# Set up logging with file handler for persistent logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level to get more detailed logs

# Clean up old log file if it exists
if log_file.exists():
    try:
        log_file.unlink()  # Delete the existing log file
        print(f"Deleted old log file: {log_file}")
    except Exception as e:
        print(f"Warning: Could not delete old log file {log_file}: {e}")

# Create a file handler to save logs to a file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Keep console handler for real-time monitoring
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class AccountManager:
    def __init__(self):
        self.accounts = []
        self.lock = threading.Lock()  # Add a lock for thread safety
        logger.info("Initializing AccountManager")

    def add_account(self, account: Account):
        with self.lock:
            self.accounts.append(account)
            logger.debug(f"Added account: {account.username}")

    def load_account(self, path: str):
        try:
            logger.debug(f"Loading account from: {path}")
            with open(path, 'r') as f:
                account = json.load(f)
                self.add_account(Account(path, **account))
                logger.info(f"Successfully loaded account: {account.get('username', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error loading account from {path}: {str(e)}", exc_info=True)
            raise

    def get_account(self):
        with self.lock:
            if not self.accounts:
                logger.warning("No accounts available")
                return None
            return_account = self.accounts.pop(0)
            # Add back to the end of the list to create a rotation
            self.accounts.append(return_account)
            logger.debug(f"Providing account: {return_account.username}")
            return return_account
    
    def return_account(self, account: Account):
        with self.lock:
            logger.debug(f"Returning account to pool: {account.username}")
            self.accounts.append(account)
    
    def load_from_dir(self, path: str = "accounts"):
        # each subdir has an account.json file
        logger.info(f"Loading accounts from directory: {path}")
        if not os.path.exists(path):
            logger.warning(f"Directory {path} does not exist")
            return
            
        for account_dir in os.listdir(path):
            account_path = os.path.join(path, account_dir, "account.json")
            if os.path.exists(account_path):
                try:
                    self.load_account(account_path)
                except Exception as e:
                    logger.error(f"Error loading account {account_path}: {str(e)}")

    def print_accounts(self):
        with self.lock:
            logger.info(f"Number of accounts: {len(self.accounts)}")
            for account in self.accounts:
                logger.info(f"Account: {account.username}")
    
    def delete_account_cookies(self, account: Account):
        # to get the cookie path, we need to replace the account.json at the end of the path with cookies.json
        cookie_path = os.path.join(os.path.dirname(account.source_file), "cookies.json")
        if os.path.exists(cookie_path):
            try:
                os.remove(cookie_path)
                logger.info(f"Deleted cookies for account: {account.username}")
            except Exception as e:
                logger.error(f"Error deleting cookies for {account.username}: {str(e)}")
    
    def clear_cookies(self):
        with self.lock:
            logger.info("Clearing cookies for all accounts")
            for account in self.accounts:
                self.delete_account_cookies(account)
    
    def get_account_count(self):
        with self.lock:
            count = len(self.accounts)
            logger.debug(f"Account count: {count}")
            return count

class AccountServer:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.account_manager = AccountManager()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = False
        logger.info(f"Initializing AccountServer on {host}:{port}")
    
    def start(self, accounts_dir="accounts"):
        logger.info(f"Starting AccountServer with accounts directory: {accounts_dir}")
        self.account_manager.load_from_dir(accounts_dir)
        
        if self.account_manager.get_account_count() == 0:
            logger.error("No accounts available, cannot start server")
            return False
        
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        logger.info(f"AccountServer running on {self.host}:{self.port}")
        
        try:
            while self.running:
                client_socket, address = self.server_socket.accept()
                logger.debug(f"New connection from: {address}")
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, address))
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
        finally:
            self.server_socket.close()
            logger.info("Server socket closed")
        
        return True
    
    def handle_client(self, client_socket, address):
        logger.debug(f"Handling client: {address}")
        try:
            # First read the command as a string
            data = client_socket.recv(1024).decode('utf-8')
            
            if data == "GET_ACCOUNT":
                logger.debug(f"GET_ACCOUNT request from {address}")
                account = self.account_manager.get_account()
                if account:
                    account_dict = account.to_dict()
                    serialized_account = pickle.dumps(account_dict)
                    client_socket.sendall(serialized_account)
                    logger.info(f"Sent account {account.username} to {address}")
                else:
                    client_socket.sendall(pickle.dumps(None))
                    logger.warning(f"No account available to send to {address}")
            
            elif data == "RETURN_ACCOUNT":
                logger.debug(f"RETURN_ACCOUNT request from {address}")
                # Wait for confirmation from client before receiving binary data
                client_socket.sendall(b"READY_FOR_DATA")
                # Now receive the binary data
                serialized_account = client_socket.recv(4096)
                account_dict = pickle.loads(serialized_account)
                if account_dict:
                    account = Account(**account_dict)
                    self.account_manager.return_account(account)
                    client_socket.sendall(b"SUCCESS")
                    logger.info(f"Account {account.username} returned by {address}")
                else:
                    client_socket.sendall(b"FAILED")
                    logger.warning(f"Received empty account return from {address}")
            
            elif data == "GET_ACCOUNT_COUNT":
                logger.debug(f"GET_ACCOUNT_COUNT request from {address}")
                count = self.account_manager.get_account_count()
                client_socket.sendall(str(count).encode('utf-8'))
                logger.debug(f"Sent account count: {count}")
            
            else:
                logger.warning(f"Unknown request from {address}: {data}")
                client_socket.sendall(b"UNKNOWN_REQUEST")
                
        except UnicodeDecodeError as e:
            logger.error(f"Decoding error from {address}. Client likely sent binary data when text was expected: {e}")
            client_socket.sendall(b"PROTOCOL_ERROR")
        except Exception as e:
            logger.error(f"Error handling client {address}: {str(e)}", exc_info=True)
            try:
                client_socket.sendall(b"ERROR")
            except:
                pass
        finally:
            client_socket.close()
    
    def stop(self):
        logger.info("Stopping AccountServer")
        self.running = False
        # Create a dummy connection to unblock accept()
        try:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.host, self.port))
            logger.debug("Sent dummy connection to unblock server")
        except:
            logger.debug("Failed to send dummy connection")

class AccountClient:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        logger.info(f"Initializing AccountClient connecting to {host}:{port}")
    
    def get_account(self):
        logger.debug("Requesting account from server")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                
                # Request an account
                s.sendall(b"GET_ACCOUNT")
                
                # Receive and deserialize the account
                serialized_account = s.recv(4096)
                account_dict = pickle.loads(serialized_account)
                
                if account_dict:
                    # Ensure path is included in the dictionary (use empty string if missing)
                    if 'path' not in account_dict:
                        account_dict['path'] = ""
                        
                    account = Account(**account_dict)
                    logger.info(f"Received account: {account.username}")
                    return account
                else:
                    logger.warning("No account received from server")
                    return None
        
        except Exception as e:
            logger.error(f"Error getting account: {str(e)}", exc_info=True)
            return None
    
    def return_account(self, account):
        if not account:
            logger.warning("Attempted to return None account")
            return False
        
        logger.debug(f"Returning account {account.username} to server")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                
                # Indicate we want to return an account
                s.sendall(b"RETURN_ACCOUNT")
                
                # Wait for server to be ready for binary data
                ready_signal = s.recv(1024)
                if ready_signal != b"READY_FOR_DATA":
                    logger.error(f"Expected READY_FOR_DATA but got: {ready_signal}")
                    return False
                
                # Serialize and send the account
                account_dict = account.to_dict()
                serialized_account = pickle.dumps(account_dict)
                s.sendall(serialized_account)
                
                # Get confirmation
                result = s.recv(1024)
                if result == b"SUCCESS":
                    logger.info(f"Successfully returned account {account.username}")
                    return True
                else:
                    logger.error(f"Failed to return account: {result.decode('utf-8', errors='replace')}")
                    return False
                
        except Exception as e:
            logger.error(f"Error returning account: {str(e)}", exc_info=True)
            return False
    
    def get_account_count(self):
        logger.debug("Requesting account count from server")
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                
                # Request account count
                s.sendall(b"GET_ACCOUNT_COUNT")
                
                # Receive the count
                count = int(s.recv(1024).decode('utf-8'))
                logger.debug(f"Received account count: {count}")
                return count
        
        except Exception as e:
            logger.error(f"Error getting account count: {str(e)}", exc_info=True)
            return 0


if __name__ == "__main__":
    server = AccountServer()
    try:
        logger.info("Starting Account Server")
        server.start()
    except KeyboardInterrupt:
        logger.info("Account Server interrupted by user")
        server.stop()
    finally:
        logger.info("Account Server shutdown complete")