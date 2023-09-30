//! Digest trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::DIGEST_SIZE;

use super::Felt;
use ark_serialize::CanonicalSerialize;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// An Anemoi Digest for the Anemoi Hash over Felt
pub struct AnemoiDigest([Felt; DIGEST_SIZE]);

impl AnemoiDigest {
    /// Returns a new Digest from a provided array
    pub fn new(value: [Felt; DIGEST_SIZE]) -> Self {
        Self(value)
    }

    /// Returns a reference to the wrapped digest
    pub fn as_elements(&self) -> &[Felt; DIGEST_SIZE] {
        &self.0
    }

    /// Returns the wrapped digest
    pub fn to_elements(&self) -> [Felt; DIGEST_SIZE] {
        self.0
    }

    /// Returns a `Vec<Felt>` from the provided digest slice
    pub fn digests_to_elements(digests: &[Self]) -> Vec<Felt> {
        let mut res = Vec::with_capacity(digests.len() * DIGEST_SIZE);
        for digest in digests {
            res.extend(digest.as_elements())
        }

        res
    }

    /// Returns an array of bytes corresponding to the digest
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut bytes = [0u8; 48];
        self.0[0].serialize_compressed(&mut bytes[..]).unwrap();
        bytes
    }
}

impl Default for AnemoiDigest {
    fn default() -> Self {
        AnemoiDigest([Felt::default(); DIGEST_SIZE])
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::Zero;
    use super::*;
    use ark_ff::UniformRand;
    use rand_core::OsRng;

    #[test]
    fn digest_elements() {
        let mut rng = OsRng;

        for _ in 0..100 {
            let mut array = [Felt::zero(); DIGEST_SIZE];
            for item in array.iter_mut() {
                *item = Felt::rand(&mut rng);
            }

            let digest = AnemoiDigest::new(array);
            assert_eq!(digest.to_elements(), array);
            assert_eq!(&digest.to_elements(), digest.as_elements());
            assert_eq!(
                digest.as_elements(),
                &AnemoiDigest::digests_to_elements(&[digest])[..]
            );
        }

        let digest = AnemoiDigest::default();
        assert_eq!(digest.to_elements(), [Felt::zero(); DIGEST_SIZE]);
        assert_eq!(digest.as_elements(), &vec![Felt::zero(); DIGEST_SIZE][..]);
        assert_eq!(digest.to_bytes(), [0u8; 48]);
    }
}
