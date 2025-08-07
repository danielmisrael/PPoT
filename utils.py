import multiprocessing
import dill
import matplotlib.pyplot as plt


def no_file_operations(*args, **kwargs):
    """Silently ignore file operations by returning a dummy file-like object"""
    class DummyFile:
        def write(self, data): pass
        def read(self): return ""
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def flush(self): pass

    return DummyFile()

def execute_and_return_figure(code_string, queue):
    """
    Executes code in a separate process, and if successful,
    serializes the matplotlib figure and puts it on the queue.
    """
    try:
        # We need a non-interactive backend for this to work reliably
        # when not running in a main thread.
        plt.switch_backend('Agg')

        # Override file operations
        restricted_globals = globals().copy()
        restricted_globals.update({
            'open': no_file_operations,
        })

        # Override builtins that could do file operations
        restricted_builtins = __builtins__.copy() if isinstance(__builtins__, dict) else __builtins__.__dict__.copy()
        restricted_builtins.update({
            'open': no_file_operations,
        })
        restricted_globals['__builtins__'] = restricted_builtins

        # A dictionary to hold the local variables from exec
        local_scope = {}

        exec(code_string, restricted_globals, local_scope)

        # Get the figure object created by the code
        fig = plt.gcf()

        # Check if a plot was actually created
        if not fig.get_axes():
            raise ValueError("The executed code did not generate a plot.")

        # Use dill to serialize the figure object into bytes
        serialized_figure = dill.dumps(fig)
        queue.put(serialized_figure)

    except Exception as e:
        # Put the error message on the queue on failure
        queue.put(e)
    finally:
        # Ensure cleanup in the subprocess
        plt.close('all')

def safe_execute_plot(code_to_run, timeout_seconds=30):

    return_queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=execute_and_return_figure,
        args=(code_to_run, return_queue)
    )
    process.start()
    process.join(timeout=timeout_seconds)

    # Check if we have a result first, even if process appears alive
    try:
        result = return_queue.get(timeout=1)

        if isinstance(result, bytes):
            figure = dill.loads(result)
            return True, figure
        elif isinstance(result, Exception):
            return False, str(result)

    except:
        pass

    # Clean up process if still running
    if process.is_alive():
        process.terminate()
        process.join()
        return False, "Process timed out."

    return False, "Process completed but no result available."

