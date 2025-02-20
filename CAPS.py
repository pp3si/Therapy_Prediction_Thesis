def ProgressBar(index, total):
    '''
    This function uses values 0 through 100 to create a progress bar.
    Make sure the index will reach the total. You may, for example, need
    to subract one from total.
    '''
    percent_no_round = index / total * 100

    def print_bar(percent):
        p = round(percent / 5)
        if percent < 100 and percent >= 0:
            return print('\r[', '\033[0;43m', ' ' * p, '\033[0m', ' ' * (20 - p), ']', percent, '%', end='', sep='', flush=True)
        if percent == 100:
            return print('\r[', '\033[0;42m', ' ' * 20, '\033[0m', ']', percent, '%', end='\n', sep='', flush=True)

    if total < 1000:
        if abs(percent_no_round % 1) <= .1:
            print_bar(round(percent_no_round))
    else:
        if (percent_no_round % 1) > (((index + 1) / total * 100) % 1):
            print_bar(round(percent_no_round))