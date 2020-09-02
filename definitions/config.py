# -*- coding: utf-8 -*-

"""
Utility functions for reading configurations.

Author: Jan Schlüter
"""

import os
import sys
import io


class TrackingDict(dict):
    """
    Dictionary subclass that tracks which keys have been accessed for reading,
    and can return a set of keys that have been set, but never read.
    """
    def __init__(self, *args, **kwargs):
        super(TrackingDict, self).__init__(*args, **kwargs)
        self.keys_read = set()

    def __getitem__(self, key):
        self.keys_read.add(key)
        return dict.__getitem__(self, key)

    def get(self, key, *args, **kwargs):
        self.keys_read.add(key)
        return dict.get(self, key, *args, **kwargs)

    @property
    def keys_not_read(self):
        return set(self.keys()) - self.keys_read


def parse_value(value):
    """
    Tries to convert `value` to a float/int/str, otherwise returns as is.
    """
    for convert in (int, float, str):
        try:
            value = convert(value)
        except ValueError:
            continue
        else:
            break
    return value


def parse_variable_assignments(assignments):
    """
    Parses a list of key=value strings and returns a corresponding dictionary.
    Values are tried to be interpreted as float or int, otherwise left as str.
    """
    variables = TrackingDict()
    for assignment in assignments or ():
        key, value = assignment.split('=', 1)
        variables[key] = parse_value(value)
    return variables


def parse_config_file(filename):
    """
    Parses a file of key=value lines and returns a corresponding dictionary.
    Values are tried to be interpreted as float or int, otherwise left as str.
    Empty lines and lines starting with '#' are ignored.
    """
    with io.open(filename, 'r') as f:
        return parse_variable_assignments(
                [l.rstrip('\r\n') for l in f
                 if l.rstrip('\r\n') and not l.startswith('#')])


def write_config_file(filename, cfg):
    """
    Writes out a dictionary of configuration flags into a text file that is
    understood by parse_config_file(). Keys are sorted alphabetically.
    """
    with io.open(filename, 'wb' if sys.version_info[0] == 2 else 'w') as f:
        f.writelines("%s=%s\n" % (key, cfg[key]) for key in sorted(cfg))


def prepare_argument_parser(parser):
    """
    Adds suitable --vars and --var arguments to an ArgumentParser instance.
    """
    parser.add_argument('--vars', metavar='FILE',
            action='append', type=str,
            default=[os.path.join(os.path.dirname(__file__), 'defaults.vars')],
            help='Reads configuration variables from a FILE of KEY=VALUE '
                 'lines. Can be given multiple times, settings from later '
                 'files overriding earlier ones. Will read defaults.vars, '
                 'then files given here.')
    parser.add_argument('--var', metavar='KEY=VALUE',
            action='append', type=str,
            help='Set the configuration variable KEY to VALUE. Overrides '
                 'settings from --vars options. Can be given multiple times.')


def from_parsed_arguments(options):
    """
    Read configuration files passed with --vars and immediate settings
    passed with --var from a given ArgumentParser namespace and returns a
    configuration dictionary.
    """
    cfg = TrackingDict()
    for fn in options.vars:
        cfg.update(parse_config_file(fn))
    cfg.update(parse_variable_assignments(options.var))
    return cfg


def add_defaults(cfg, filename='defaults.vars', pyfile=None):
    """
    Reads a configuration file and adds all settings to the given dictionary
    that are not defined yet. The file name of the configuration file can
    be specified completely, or will be taken to be relative to the location
    of a given Python script file.
    """
    if pyfile is not None:
        filename = os.path.join(os.path.dirname(pyfile), filename)
    defaults = parse_config_file(filename)
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v


def renamed_prefix(cfg, old, new):
    """
    Returns a new dictionary that has all keys of `cfg` that begin with `old`
    renamed to begin with `new` instead.
    """
    renamed = dict(cfg)
    for k in cfg.keys():
        if k.startswith(old):
            renamed[new + k[len(old):]] = renamed.pop(k)
    return renamed


def warn_unused_variables(cfg, expected=()):
    """
    Prints a warning about configuration variables that have never been read.
    Accepts an iterable of variables that are expected not to be read (yet).
    """
    unused = cfg.keys_not_read - set(expected)
    if unused:
        print("Warning: The following configuration variables were not used "
              "so far. Make sure you did not misspell them: " +
              ", ".join(sorted(unused)))
