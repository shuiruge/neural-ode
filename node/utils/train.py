import tensorflow as tf


def print_status_bar(process_ratio,
                     metrics=None,
                     process_bar_len=50,
                     stop=False):
    """Prints the status bar while training.

    Args:
        process_ratio: float
        metrics: Optional[List[Tuple[str, tf.Tensor]]]
            List of pairs `(metric_name, metric_value)`
        process_bar_len: int
        stop: bool
    """
    process = int(process_ratio * process_bar_len) + 1
    process_bar = '[{0}>{1}]'.format('=' * (process - 1),
                                     '-' * (process_bar_len - process))

    metrics = [] if metrics is None else metrics

    print_args = ['\r', process_bar]
    for name, value in metrics:
        print_args += [' - ', name, value]

    if stop:
        tf.print(*print_args, end='\n')
    else:
        tf.print(*print_args, end='')
