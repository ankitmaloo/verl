import os
import json
import random
import string
import re
import requests
from typing import List, Dict, Optional, Set
from datetime import datetime
import json


SCHEMA = {
    "type": "json_schema",
    "strict":True,
    "name": "AnswerAndDate",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The concise answer or summary."
                },
                "date": {
                    "type": "string",
                    "format": "date",
                    "description": "Today's date in YYYY-MM-DD."
                }
            },
            "required": ["answer", "date"]
        }
}

def ask_with_search(prompt: str,  model: str = "gpt-4.1", temperature: float = 0.1) -> dict:
    """
    Generic function that calls OpenAI Responses API with web search enabled.

    Args:
        prompt: The search query/prompt
        schema: The JSON schema for response format
        model: OpenAI model to use
        temperature: Temperature for response generation

    Returns:
        dict: Parsed JSON response based on provided schema
    """
    response = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": SCHEMA
        },
        tools=[
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate"
                },
                "search_context_size": "medium"
            }
        ],
        temperature=temperature,
        top_p=1,
        store=True,
        include=["web_search_call.action.sources"]
    )
    return json.loads(response.output_text)

def utils_get_wordle() -> dict:
    """
    Gets today's Wordle answer using web search.
    Returns: { "answer": str, "date": "YYYY-MM-DD" }
    """
    #prompt = "What is today's wordle answer? In your answer only include the word, no other text"
    return "proxy" #ask_with_search(prompt,temperature=1)

def utils_get_emoji() -> dict:
    """
    Gets today's actual moon phase emoji using web search.
    Returns: { "emoji": str, "date": "YYYY-MM-DD" }
    """
    #prompt = "What is today's current moon phase? Return only the appropriate emoji: 🌑 🌒 🌓 🌔 🌕 🌖 🌗 🌘"
    return "🌕" #ask_with_search(prompt, temperature=0.1)


def create_captcha():
    """Generates a random five-character combination of 3 letters and 2 numbers."""
    letters = [random.choice(string.ascii_letters) for _ in range(3)]
    numbers = [random.choice(string.digits) for _ in range(2)]
    captcha_list = letters + numbers
    random.shuffle(captcha_list)
    captcha = ''.join(captcha_list)
    return captcha.lower()

countries = [
    "Albania", "Algeria", "Andorra", "Armenia", "Austria", "Bahrain", "Belarus",
    "Belgium", "Bolivia", "Burundi", "Comoros", "Croatia", "Czechia", "Denmark",
    "Ecuador", "Eritrea", "Estonia", "Finland", "Georgia", "Germany", "Grenada",
    "Hungary", "Iceland", "Ireland", "Jamaica", "Lebanon", "Lesotho", "Liberia",
    "Moldova", "Morocco", "Myanmar", "Namibia", "Nigeria", "Romania", "Senegal",
    "Somalia", "Tunisia", "Ukraine", "Uruguay", "Vanuatu", "Vietnam"
]

def random_country():
    """Gives options of random seven letter countries for the game."""
    return random.choice(countries)

def get_wordle_answer():
    """Fetch today's Wordle answer from web."""
    try:
        response = utils_get_wordle()
        if response and "answer" in response:
            return response["answer"].lower()
    except:
        pass
    return "crane"  # Fallback answer

def get_current_moon_phase():
    """Get current moon phase emoji using OpenAI + search."""
    try:
        #result = utils_get_emoji()
        #return result["answer"]
        return "🌕"
    except:
        raise Exception("Cannot determine real moon phase - API unavailable")

rules = [
    "Your password must be at least 5 characters.",
    "Your password must include a number.",
    "Your password must include an uppercase letter.",
    "Your password must include a special character.",
    "The digits in your password must add up to 25.",
    "Your password must include a month of the year.",
    "Your password must include a roman numeral.",
    "Your password must include one of our sponsors: (Pepsi, Starbucks, Shell) - case insensitive",
    "The roman numerals in your password should multiply to 35.",
    "Your password must include this CAPTCHA: {captcha}",
    "Your password must include today's Wordle answer.",
    "Your password must include a two letter symbol from the periodic table.",
    "Your password must include the current phase of the moon as an emoji",
    "Your password must include the name of this country: {country}",
    "Your password must include a leap year.",
    "🥚 ← This is my chicken Paul. He hasn't hatched yet, please put him in your password and keep him safe.",
    "The elements in your password must have atomic numbers that add up to 200.",
    "Your password is not strong enough 🏋️‍♂️",
    "Your password must contain one of the following affirmations: (I am loved, I am worthy, I am enough)",
    "Paul has hatched! Please don't forget to feed him, he eats three 🐛.",
    "A sacrifice must be made. Pick two letters that you will no longer be able to use.",
    "Your password must include this color in hex. green",
    "Your password must include the length of your password.",
    "The length of your password must be a prime number.",
    "Uh lets skip this one",
    "Your password must contain three consecutive letters that appear in alphabetical order (like 'abc', 'hij', 'xyz')"
]

instructions = """
You - the assistant - are playing a password game. After each step, you will be given a rule. The password you submit has to comply with that rule and all previous rules.
Only submit as final answer in the specified format in your answer.

GOAL: When given a rule, in your answer, only generate the password string that satisfies the rules.

Some clarifications as you may need them:
1. Only capital letters count towards Roman Numbers - if it's a valid roman number.
2. For a substring to count as an element, first (or only) letter of the symbol has to be capitalized.
3. For other answers, you can play around with capitalization or not to satisfy the given rules. Checks are case insensitive othersie - not for roman or elements.
4. Try to play as long as you can. 

RESPONSE GUIDELINES:
Answer should be in markdown in following format:
### Password
{password}
### Giveup
{True if you want to give up, False otherwise}
"""

class PasswordGame:
    def __init__(self):
        self.current_rule = 0
        self.game_active = True
        self.captcha = create_captcha()
        self.country = random_country()
        self.wordle_answer = get_wordle_answer()
        self.moon_phase = get_current_moon_phase()
        self.password_history = []


    def get_current_rule(self) -> Optional[str]:
        if self.current_rule >= len(rules) or not self.game_active:
            return None
        rule = rules[self.current_rule]
        if "{captcha}" in rule:
            return rule.format(captcha=self.captcha)
        elif "{country}" in rule:
            return rule.format(country=self.country)
        return rule

    def get_all_rules_up_to_current(self) -> List[str]:
        formatted_rules = []
        for i, rule in enumerate(rules[:self.current_rule + 1]):
            if "{captcha}" in rule:
                formatted_rules.append(rule.format(captcha=self.captcha))
            elif "{country}" in rule:
                formatted_rules.append(rule.format(country=self.country))
            else:
                formatted_rules.append(rule)
        return formatted_rules

    def advance_rule(self):
        self.current_rule += 1
        if self.current_rule >= len(rules):
            self.game_active = False

    def end_game(self):
        self.game_active = False

    def calculate_reward(self, password: str) -> float:
        """Calculate reward: +1 per passing rule, -0.1 per character."""
        satisfied_rules = 0

        # Check all rules up to current rule (inclusive when game ends)
        rule_count = self.current_rule if self.game_active else len(rules)

        for i in range(rule_count):
            if self._check_rule(password, i):
                satisfied_rules += 1

        # +1 per passing rule, -0.1 per character
        rule_score = satisfied_rules
        length_penalty = len(password) * 0.1
        total_reward = rule_score - length_penalty

        return round(total_reward, 1)

    def get_rule_feedback(self, password: str) -> Dict:
        """Get detailed feedback on which rules pass/fail."""
        feedback = {
            "password": password,
            "length": len(password),
            "rules_checked": [],
            "total_passing": 0,
            "reward": 0.0
        }

        # For feedback, include current rule if game is active
        rule_count = (self.current_rule + 1) if self.game_active else len(rules)

        for i in range(rule_count):
            passes = self._check_rule(password, i)
            rule_text = rules[i]
            if "{captcha}" in rule_text:
                rule_text = rule_text.format(captcha=self.captcha)
            elif "{country}" in rule_text:
                rule_text = rule_text.format(country=self.country)

            feedback["rules_checked"].append({
                "rule_index": i,
                "rule_text": rule_text,
                "passes": passes
            })
            if passes:
                feedback["total_passing"] += 1

        feedback["reward"] = self.calculate_reward(password)
        return feedback

    def _check_rule(self, password: str, rule_index: int) -> bool:
        """Comprehensive rule checking for all password rules."""
        if rule_index == 0:  # At least 5 characters
            return len(password) >= 5

        elif rule_index == 1:  # Include a number
            return any(c.isdigit() for c in password)

        elif rule_index == 2:  # Include uppercase letter
            return any(c.isupper() for c in password)

        elif rule_index == 3:  # Include special character
            return any(not c.isalnum() for c in password)

        elif rule_index == 4:  # Digits sum to 25
            digit_sum = sum(int(c) for c in password if c.isdigit())
            return digit_sum == 25

        elif rule_index == 5:  # Include month
            months = ['january', 'february', 'march', 'april', 'may', 'june',
                     'july', 'august', 'september', 'october', 'november', 'december']
            return any(month in password.lower() for month in months)

        elif rule_index == 6:  # Include roman numeral
            roman_pattern = r'[IVXLCDM]+'
            return bool(re.search(roman_pattern, password))

        elif rule_index == 7:  # Include sponsor
            sponsors = ['pepsi', 'starbucks', 'shell']
            return any(sponsor in password.lower() for sponsor in sponsors)

        elif rule_index == 8:  # Roman numerals multiply to 35
            return self._check_roman_multiply(password, 35)

        elif rule_index == 9:  # Include CAPTCHA
            return self.captcha in password.lower()

        elif rule_index == 10:  # Include Wordle answer
            return self.wordle_answer.lower() in password.lower()

        elif rule_index == 11:  # Include periodic element
            return self._check_periodic_element(password)

        elif rule_index == 12:  # Include moon phase emoji
            return self.moon_phase in password

        elif rule_index == 13:  # Include country (dynamic)
            return self.country.lower() in password.lower()

        elif rule_index == 14:  # Include leap year
            return self._check_leap_year(password)

        elif rule_index == 15:  # Include Paul (egg emoji)
            return '🥚' in password

        elif rule_index == 16:  # Atomic numbers sum to 200
            return self._check_atomic_sum(password, 200)

        elif rule_index == 17:  # Not strong enough (always fails)
            return False

        elif rule_index == 18:  # Include affirmation
            affirmations = ['i am loved', 'i am worthy', 'i am enough']
            return any(affirmation in password.lower() for affirmation in affirmations)

        elif rule_index == 19:  # Paul eats 3 bugs
            return password.count('🐛') == 3

        elif rule_index == 20:  # Sacrifice two letters
            # This would need state tracking - simplified for now
            return True

        elif rule_index == 21:  # Include green hex
            green_hex_pattern = r'#00[89ab]000|#008000'
            return bool(re.search(green_hex_pattern, password.lower()))

        elif rule_index == 22:  # Include password length
            return str(len(password)) in password

        elif rule_index == 23:  # Length is prime
            return self._is_prime(len(password))

        elif rule_index == 24:  # Skip this one
            return True

        elif rule_index == 25:  # Include 3 consecutive chars
            # Normalize to lowercase for case-insensitive checking
            pwd = password.lower()

            for i in range(len(pwd) - 2):
                triplet = pwd[i:i+3]

                # Must be three letters
                if triplet.isalpha():
                    # Check if they're consecutive: a->b->c, etc.
                    if ord(triplet[2]) - ord(triplet[0]) == 2:
                        # Verify the middle character is exactly +1
                        if ord(triplet[1]) - ord(triplet[0]) == 1:
                            return True

            return False

    def get_game_state(self) -> Dict:
        return {
            "current_rule_index": self.current_rule,
            "current_rule": self.get_current_rule(),
            "all_rules": self.get_all_rules_up_to_current(),
            "game_active": self.game_active,
            "instructions": instructions,
            "captcha": self.captcha,
            "country": self.country,
            "wordle_answer": self.wordle_answer,
            "moon_phase": self.moon_phase
        }

    def get_instructions(self) -> str:
        return instructions

    def get_minimal_game_state(self) -> Dict:
        """Return minimal game state, only exposing non-searchable values when needed."""
        state = {
            "current_rule_index": self.current_rule,
            "current_rule": self.get_current_rule(),
            "all_rules": self.get_all_rules_up_to_current(),
            "game_active": self.game_active
        }

        # Only expose captcha when rule 9 (index 9) is active or passed
        if self.current_rule >= 9:
            state["captcha"] = self.captcha

        # Only expose country when rule 13 (index 13) is active or passed
        if self.current_rule >= 13:
            state["country"] = self.country

        return state

    def _check_roman_multiply(self, password: str, target: int) -> bool:
        """Check if roman numerals in password multiply to target."""
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        roman_pattern = r'[IVXLCDM]+'
        romans = re.findall(roman_pattern, password)

        if not romans:
            return False

        product = 1
        for roman in romans:
            value = self._roman_to_int(roman)
            if value > 0:
                product *= value

        return product == target

    def _roman_to_int(self, roman: str) -> int:
        """Convert roman numeral to integer."""
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0

        for char in reversed(roman):
            value = roman_values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value

        return total

    def _check_periodic_element(self, password: str) -> bool:
        """Check for periodic table elements (first letter capitalized)."""
        elements = [
            'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
            'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
        return any(element in password for element in elements)

    def _check_leap_year(self, password: str) -> bool:
        """Check for leap years in password."""
        numbers = re.findall(r'\d{4}', password)
        for num_str in numbers:
            year = int(num_str)
            if self._is_leap_year(year):
                return True
        return False

    def _is_leap_year(self, year: int) -> bool:
        """Check if year is leap year."""
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def _check_atomic_sum(self, password: str, target: int) -> bool:
        """Check if atomic numbers of elements sum to target."""
        element_atomic = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
            'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
            'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
        }

        total_atomic = 0
        for element, atomic_num in element_atomic.items():
            if element in password:
                total_atomic += atomic_num

        return total_atomic == target

    def step(self, password: str=None, give_up:bool=False):
        """
        The main interaction function for the RL environment.
        You submit a password, and it returns the new state.
        """

        if give_up:
          self.end_game()
          reward = self.calculate_reward(password)
          feedback = self.get_rule_feedback(password)
          return {
            "game_ended": True,
            "gave_up": True,
            "reward": reward,
            "final_password": password,
            "rule_feedback": feedback
          }
        if password is not None:
          self.password_history.append(password)

        if len(self.password_history) == 0:
          return {"current_rule_index": self.current_rule,
          "current_rule": self.get_current_rule(),
          "game_active": self.game_active,
          "instructions": self.get_instructions(),
          }

        # Advance to next rule
        self.advance_rule()

        # Check if game ended naturally
        if not self.game_active:
          reward = self.calculate_reward(password)
          feedback = self.get_rule_feedback(password)
          return {
            "game_ended": True,
            "gave_up": False,
            "reward": reward,
            "final_password": password,
            "rule_feedback": feedback
          }

        return self.get_minimal_game_state()


    def _is_prime(self, n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    


### Utils 

def parse_resp(resp):
    pattern = r"### Password\s*\n([\s\S]+?)\s*\n### Giveup\s*\n(true|false)"

    # Use re.IGNORECASE to match "True", "False", "true", or "false"
    match = re.search(pattern, resp, re.IGNORECASE)
    if match:
        # Group 1 is the password
        password = match.group(1).strip()

        # Group 2 is the giveup string ("true" or "false")
        # We convert it to a Python boolean
        giveup_str = match.group(2).lower()
        giveup_boolean = giveup_str == 'true'

        return password, giveup_boolean
    else:
        # Return None if the pattern doesn't match the response
        return None, None

