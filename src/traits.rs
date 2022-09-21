use ark_ff::Field;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Trait for implementing a Sponge construction.
pub trait Sponge<F: Field> {
    /// Specifies a digest type returned by this hasher.
    type Digest;

    /// Returns a hash of the provided sequence of bytes.
    fn hash(bytes: &[u8]) -> Self::Digest;

    /// Returns a hash of the provided sequence of field elements.
    fn hash_field(elems: &[F]) -> Self::Digest;

    /// Compresses two given digests into one.
    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest;
}

/// Trait for implementing a Jive compression function instantiation.
pub trait Jive<F: Field> {
    /// Compresses the provided field element slice as input by 2.
    ///
    /// The slice must be of the same length than the underlying hash state.
    fn compress(elems: &[F]) -> Vec<F>;

    /// Compresses the provided field element slice as input by a factor k.
    ///
    /// The slice must be of the same length than the underlying hash state.
    fn compress_k(elems: &[F], k: usize) -> Vec<F>;
}
