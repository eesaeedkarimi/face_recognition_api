import datetime
import re


# Adding time to input string
def add_time(request_id):
    """
    Add time to input string.

    Parameters
    ----------
    request_id : str
        Input text

    Returns
    -------
    str
        Updated text by adding current time to it
    """

    now = datetime.datetime.now()
    now = f'-{now.hour}-{now.minute}-{now.second}-{now.microsecond // 1000}'
    request_id += now

    return request_id


# Mapping the error line to the proper error code
def error_line2error_code(error_line):
    """
    Map the error line to the proper error code.

    Parameters
    ----------
    error_line : str
        An string that describe the error

    Returns
    -------
    int
        Error code which match the error line
    """

    for key in ERROR_CODES.keys():
        if key in error_line:
            return str(ERROR_CODES[key])

    return error_line


def prepare_the_error_message(logger_model, results, error_message, status_code, message_title=''):
    """
    This function is responsible for generating an error message based on the input information

    Parameters
    ----------
    logger_model : object
        A global model for logger
    results : dict
         A dictionary that contains the our result information
    error_message : list
        List of error messages
    status_code : int
        Status code of the error message
    message_title : str
        A message title for the error message

    Returns
    -------
    error_info : dict
        Generated error message
    status code : int
        Status code of the error message
    """

    results['error_message'].extend(error_message)
    log_message = ', '.join(results['error_message'])
    logger_model.error(log_message)

    return {'error_message': f'{message_title}' + f'{log_message}'}, status_code


# Removing internal errors from an error message
def remove_internal_errors(error_line):
    """
    Remove internal errors from an error message.

    Parameters
    ----------
    error_line : str
        Input error line

    Returns
    -------
    str
        Updated error line by removing internal errors
    """

    starts_iter = re.compile(r'<').finditer(error_line)
    ends_iter = re.compile(r'>: ').finditer(error_line)
    starts = [start.span()[0] for start in starts_iter]
    if not starts:
        return error_line

    ends = [end.span()[1] for end in ends_iter]
    starts[-1] -= 2
    ends.insert(0, 0)
    ends.pop(-1)
    output_error_line = ''.join([error_line[end:start] for end, start in zip(ends, starts)])

    return output_error_line
