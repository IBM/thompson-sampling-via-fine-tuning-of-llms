import builtins, threading, re
from typing import Callable, Any
from RestrictedPython import safe_builtins, limited_builtins, utility_builtins
#from RestrictedPython import compile_restricted

def _timeout_handler() -> None:
    """
        Raises an Exception due to a time out in the code that is run.
    """
    raise Exception("The operation timed out!")

def _custom_range(*args, start=0, stop=None, step=1):
    """
        Identical in behaviour to the builtin `range`, but due to being a custom function, `RestrictedPython` does not limit the loop size to 5'000.
    """
    # Handle positional arguments
    if len(args) == 1:
        stop = args[0]  # Equivalent to range(stop)
    elif len(args) == 2:
        start, stop = args  # Equivalent to range(start, stop)
    elif len(args) == 3:
        start, stop, step = args  # Equivalent to range(start, stop, step)
    elif len(args) > 3:
        raise TypeError("_custom_range() expected at most 3 positional arguments, got {}".format(len(args)))    
    # Ensure stop is provided
    if stop is None:
        raise TypeError("_custom_range() missing 1 required argument: 'stop'")
    # Use a generator to yield values
    while start < stop:
        yield start
        start += step

_additional_safe_operations = {
    'abs': abs,
    'divmod': divmod,
    'pow': pow,
    'round': round,
    'sum': sum,
    'max': max,
    'min': min,
    'len': len,
    'sorted': sorted,
    'zip': zip,
    'reversed': reversed,
    'enumerate': enumerate,
    'map': map,
    'filter': filter,
    'all': all,
    'any': any,
    'iter': iter,
    'next': next,
    'list': list,
    'bool': bool,
    'int': int,
    'float': float,
    'set': set,
    'tuple': tuple,
    'dict': dict,
    'True': True,
    'False': False,
    'None': None,
}

def execute_restricted_function(func:Callable, 
                                func_arguments:dict, 
                                max_seconds:float=10) -> Any:
    """
        Executes a restricted function that takes `func_arguments` as arguments. Disables I/O operations as well as print statements
        and allows for `max_seconds` seconds before interrupting the execution through raising of an exception.

        Args:
            func (Callable): The function to be executed.
            func_arguments (dict): The arguments to be passed to the function, formatted as a dictionary that is unpacked.
            max_seconds (float, optional): The hard limit in seconds on the duration of the function execution. Defaults to 10.

        Returns:
            out (Any): The result of calling func(**func_arguments)
    """
    original_open = builtins.open
    original_print = builtins.print
    builtins.open = lambda *args, **kwargs: None # disables I/O operations
    builtins.print = lambda *args, **kwargs: None # disables printing
    result = None
    exception = None
    try:
        timer = threading.Timer(max_seconds, _timeout_handler)
        timer.start() # after running for max_seconds seconds, times out to avoid evaluating infinite loops and unneccessarily slow programs
        result = func(**func_arguments)
    except Exception as e:
        exception = e
    finally:
        timer.cancel()
        builtins.open = original_open
        builtins.print = original_print
        if exception is not None:
            raise exception
    return result

def compile_restricted_function(func_code: str, 
                                func_name:str, 
                                restricted_globals:dict) -> Callable:
    """
        Compiles the code provided in the string `func_code` and returns the function therein named `func_name`.
        The global variables `restricted_globals` can be used to allow imports and access/change certain global variables.
        Disables I/O operations as well as print statements during compilation of func_code.

        Args:
            func_code (str): The code containing the function `func_name` that is compiled.
            func_name (str): The name of a function in `func_code` that is returned upon compilation.
            restricted_globals (dict): Global variables to which the compiled func_code has access to. Can be used to enable imports or to access/change certain global variables.
        Returns:
            out (Callable): The compiled function that can be executed safely using `execute_restricted_function`.
    """
    original_open = builtins.open
    original_print = builtins.print
    builtins.open = lambda *args, **kwargs: None # disables I/O operations
    builtins.print = lambda *args, **kwargs: None # disables printing 
    exception = None
    try:
        #byte_code = compile_restricted(func_code, '<string>', 'exec') # with this the LLM cannot even perform simple numpy array indexing assignments, so we do not safely compile. At least printing and I/O operations were disabled though and only local variables + a small set of global variables can be accessed.
        byte_code = compile(func_code, '<string>', 'exec')
        restricted_globals = {**restricted_globals}
        restricted_globals.update(safe_builtins)
        restricted_globals.update(limited_builtins)
        restricted_globals.update(utility_builtins)
        restricted_globals.update(_additional_safe_operations) 
        restricted_globals['range'] = _custom_range # allows long ranges (> 1000, which restricted python would not allow)
        restricted_locals = {}
        exec(byte_code, restricted_globals, restricted_locals)
        if func_name not in restricted_locals:
            raise ValueError(f"The provided code must define a function named '{func_name}'. However, restricted_locals = {restricted_locals}")
    except Exception as e:
        exception = e
    finally: 
        builtins.open = original_open
        builtins.print = original_print
        if exception is not None:
            raise exception
    return restricted_locals[func_name]

def extract_code(code_str:str) -> str:
    """
        Extracts code from inside triple backticks (```), i.e., from markdown format. Acts as NOP if no triple backticks are present.

        Args:
            code_str (str): The string inside which the code resides.
        Returns:
            out (str): Either the part of the string insides triple backticks (```) or the full string if not triple backticks are present.
    """
    m = re.search(r'\`\`\`(.+?)\`\`\`', code_str, re.DOTALL)
    if m:
        code_str = m.group(1) 
    return code_str

def indent_multiline_string(multiline_string, desired_indent=4):
    """
    Normalizes the indentation of a multiline string such that the first non-empty line
    has a specified number of leading whitespaces, and all other lines maintain
    their relative indentation to that first line. Also removes empty lines from the string.

    Args:
        multiline_string (str): The input multiline string.
        desired_indent (int): The desired number of whitespaces for the first line's indentation.

    Returns:
        str: The re-indented multiline string.
    """
    lines = multiline_string.splitlines()
    if not lines:
        return ""
    # Find the first non-empty line and its original indentation
    first_line_indent = 0
    first_non_empty_line_found = False
    for line in lines:
        if line.strip():
            first_line_indent = len(line) - len(line.lstrip())
            first_non_empty_line_found = True
            break
    if not first_non_empty_line_found: # All lines are empty or whitespace
        return multiline_string
    # Calculate the adjustment needed for all lines
    indent_adjustment = desired_indent - first_line_indent
    new_lines = []
    for line in lines:
        if line.strip():
            original_line_indent = len(line) - len(line.lstrip())
            # Apply the adjustment to the current line's indentation
            new_indent = max(0, original_line_indent + indent_adjustment)
            new_lines.append(" " * new_indent + line.lstrip())
    return "\n".join(new_lines)