class EquationParser:
    def __init__(self):
        self.operators = {'+', '-', '*', '/', '='}
        
    def parse_equation(self, symbols):
        """Parse the equation and validate its format."""
        equation = ''.join(symbols)
        print(f"Parsed equation: {equation}")
        
        if '=' in equation:
            parts = equation.split('=')
            if len(parts) != 2:
                return None, "Invalid equation format"
            try:
                left_side = eval(parts[0])
                right_side = eval(parts[1])
                return left_side == right_side, f"{equation} is {'correct' if left_side == right_side else 'incorrect'}"
            except Exception as e:
                return None, "Invalid equation"
        else:
            try:
                result = eval(equation)
                return True, f"{equation} = {result}"
            except Exception as e:
                return None, "Invalid equation format"

