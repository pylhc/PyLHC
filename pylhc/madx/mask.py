import re 

MASK_ENDING = '.mask'


def replace_string_in_text(text, toreplace, replacedby):
        replaced_text = re.sub(toreplace, replacedby, text)
        return replaced_text


def create_madx_from_mask(maskfile, replace_keys, replace_values):

    with open(maskfile, 'r') as mfile:
        template = open(maskfile, 'r').read()

    jobname = re.sub(MASK_ENDING, '', maskfile)
    jobs = []

    for idx, values in enumerate(replace_values):
        text = template
        with open(f'{jobname}.{idx}', 'w') as madxjob:
            for i, key in enumerate(replace_keys):
                text = replace_string_in_text(text, key, f'{values[i]}')
            madxjob.write(text)
        jobs.append(f'{jobname}.{idx}')
    return jobs
