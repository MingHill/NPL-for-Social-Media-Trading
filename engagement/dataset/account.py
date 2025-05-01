"""
Account class definition used by account_service.py and client applications
"""

class Account:
    def __init__(self, path: str, username: str, password: str, email: str, user_agent: str, state: str = "active"):
        self.path = path
        self.username = username
        self.password = password
        self.email = email
        self.user_agent = user_agent
        self.state = state

    def get_cookies_path(self):
        return self.path.replace("account.json", "cookies.json")

    def get_account_path(self):
        return self.path
    
    def to_dict(self):
        return {
            "path": self.path,
            "username": self.username,
            "password": self.password,
            "email": self.email,
            "user_agent": self.user_agent,
            "state": self.state
        }

    def __str__(self):
        return f"Path: {self.path}\nUsername: {self.username}\nPassword: {self.password}\nEmail: {self.email}\nUser Agent:  {self.user_agent}"

    # These methods help with pickling/unpickling
    def __getstate__(self):
        # Return state to be pickled - the default implementation should work
        return self.__dict__

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state) 