#
# PyBroMo - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module contains utility functions to print the content of
pytables HDF5 files.
"""


def print_attrs(data_file, node_name='/', which='user', compress=False):
    """Print the HDF5 attributes for `node_name`.

    Parameters:
        data_file (pytables HDF5 file object): the data file to print
        node_name (string): name of the path inside the file to be printed.
            Can be either a group or a leaf-node. Default: '/', the root node.
        which (string): Valid values are 'user' for user-defined attributes,
            'sys' for pytables-specific attributes and 'all' to print both
            groups of attributes. Default 'user'.
        compress (bool): if True displays at most a line for each attribute.
            Default False.
    """
    node = data_file.get_node(node_name)
    print ('List of attributes for:\n  %s\n' % node)
    for attr in node._v_attrs._f_list():
        print ('\t%s' % attr)
        attr_content = repr(node._v_attrs[attr])
        if compress:
            attr_content = attr_content.split('\n')[0]
        print ("\t    %s" % attr_content)

def print_children(data_file, group='/'):
    """Print all the sub-groups in `group` and leaf-nodes children of `group`.

    Parameters:
        data_file (pytables HDF5 file object): the data file to print
        group (string): path name of the group to be printed.
            Default: '/', the root node.
    """
    base = data_file.get_node(group)
    print ('Groups in:\n  %s\n' % base)

    for node in base._f_walk_groups():
        if node is not base:
            print ('    %s' % node)

    print ('\nLeaf-nodes in %s:' % group)
    for node in base._v_leaves.itervalues():
        info = node.shape
        if len(info) == 0:
            info = node.read()
        print ('\t%s, %s' % (node.name, info))
        if len(node.title) > 0:
            print ('\t    %s' % node.title)
