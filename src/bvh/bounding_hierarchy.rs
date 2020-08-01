/// Describes a shape as referenced by a [`BoundingHierarchy`] leaf node.
/// Knows the index of the node in the [`BoundingHierarchy`] it is in.
///
/// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
///
pub trait BHShape {
    /// Sets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
    ///
    fn set_bh_node_index(&mut self, _: usize);

    /// Gets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
    ///
    fn bh_node_index(&self) -> usize;
}
