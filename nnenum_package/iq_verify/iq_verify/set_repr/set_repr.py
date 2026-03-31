import abc


class SetRepr(abc.ABC):
    def __init__(self, n_dims, discrete=None):
        # The number of dimensions this stateset occupies in euclidean space.
        self.n_dims = n_dims

        # A dictionary containing values for any other discrete variables used
        # by the stateset in reachability or simulation.
        self.properties = discrete
        if self.properties is None:
            self.properties = {}

    @abc.abstractmethod
    def plot(self, color):
        """Plot the set"""
        return

    '''
    @abc.abstractmethod
    def maximize(self, dir_vec):
        """Return the set's extreme point in the given direction"""
        return
    '''

    '''
    @abc.abstractmethod
    def get_verts(self):
        """Return the vertices of the set's convex hull"""
        return
    '''

    @abc.abstractmethod
    def affine_transform(self, A, b, in_place=False):
        """Return this set after the affine transformation"""
        return

    @abc.abstractmethod
    def split_on_halfspace_constraint(self, lhs, rhs):
        """Return the (greater_than, less_than) portions of this stateset with respect to the constraint"""
        return

    @abc.abstractmethod
    def get_bounding_box(self):
        """Returning a bounding box that contains this stateset"""
        return

    @abc.abstractmethod
    def check_is_feasible(self):
        """Return True iff this set is "feasible," e.g. has satisfiable constraints."""
        return

    @abc.abstractmethod
    def __str__(self):
        """Return a string representation of this set"""
        return

    '''
    @abc.abstractmethod
    def copy(self):
        """Return a copy of this set"""
        return
    '''

    @abc.abstractmethod
    def check_for_intersection(self, other):
        """Return True if and only if the two sets intersect"""
        return

    @abc.abstractmethod
    def intersect(self, other):
        """TODO"""
        return

    @abc.abstractmethod
    def set_difference(self, other):
        """Return this set minus other set; if there is no intersection, return this set"""

        return

    @abc.abstractmethod
    def undo_all_transformations(self):
        """TODO"""
        return

    @abc.abstractmethod
    def get_chebyshev_center(self):
        """return radius, center"""
        # (TODO: Should we return only center, and rename to 'get_witness_point'?
        return
