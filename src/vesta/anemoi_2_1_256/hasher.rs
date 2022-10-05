//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, STATE_WIDTH};
use super::{Jive, Sponge};

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// An Anemoi hash instantiation
pub struct AnemoiHash {
    state: [Felt; STATE_WIDTH],
    idx: usize,
}

impl Default for AnemoiHash {
    fn default() -> Self {
        Self {
            state: [Felt::zero(); STATE_WIDTH],
            idx: 0,
        }
    }
}

impl Sponge<Felt> for AnemoiHash {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 32];
        for (i, chunk) in bytes.chunks(31).enumerate() {
            if i < num_elements - 1 {
                buf[0..31].copy_from_slice(chunk);
            } else {
                // The last chunk may be smaller than the others, which requires a special handling.
                // In this case, we also append a byte set to 1 to the end of the string, padding the
                // sequence in a way that adding additional trailing zeros will yield a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 32];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 31 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[0] += Felt::read(&buf[..]).unwrap();
            apply_permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        Self::Digest::new(digest_array)
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            apply_permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        Self::Digest::new(digest_array)
    }

    // This will require 2 calls of the underlying Anemoi permutation.
    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        Self::hash_field(&Self::Digest::digests_to_elements(digests))
    }
}

impl Jive<Felt> for AnemoiHash {
    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress(_elems: &[Felt]) -> Vec<Felt> {
        unimplemented!()
    }

    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress_k(_elems: &[Felt], _k: usize) -> Vec<Felt> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x86acb4765790d130,
                0xecd9220a94ddd256,
                0x602258f273731322,
                0x21a6c47813fd133b,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x64c015cb450476ef,
                    0x3f0390b625b62260,
                    0x755ee12fe47ad7c6,
                    0x1a60b28355339f60,
                ])),
                Felt::new(BigInteger256([
                    0x9be33e56582dc7f2,
                    0x1683f3842cf2b526,
                    0xf2d4a4e6937d3ab2,
                    0x18e1032da17945e8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa4c208cc14cd01e3,
                    0x3923114438a0ab51,
                    0xeb55f9be10cbadfb,
                    0x13e5f7cc986f59e8,
                ])),
                Felt::new(BigInteger256([
                    0x2f345f15e1185748,
                    0x00ff161538d24e5f,
                    0x95927508bed2d48a,
                    0x0187a3786d8c6783,
                ])),
                Felt::new(BigInteger256([
                    0xcd862d0d9162ad9b,
                    0xbd38481a77f094df,
                    0xa830735362c27484,
                    0x35566f3b0afd3cc9,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb6135fec2e46056d,
                    0xd8be3701a021b20f,
                    0x91d66a0d5074c55f,
                    0x216abc69b2bb6d27,
                ])),
                Felt::new(BigInteger256([
                    0xb204e3374c989c5a,
                    0xebeaebb95d1d4068,
                    0x4ad2c18494d3f0fa,
                    0x0ebeb96419f066c1,
                ])),
                Felt::new(BigInteger256([
                    0xc267a7d93260e52e,
                    0x4400a0a0dad7422a,
                    0xed446fc34468b041,
                    0x036077abf159a2cf,
                ])),
                Felt::new(BigInteger256([
                    0xc57e42587100b2f5,
                    0x5826d6c4b2e4d171,
                    0x9e30a071637da1a9,
                    0x23d3362cdae991f5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x978e115bb0c1c28f,
                    0xe7e0e2d339a9face,
                    0xea5a84f6f631f441,
                    0x148537732e9f7acf,
                ])),
                Felt::new(BigInteger256([
                    0x790b95b3024bdc83,
                    0xc0e628a7b9c0493e,
                    0xa1867d0a2dc48269,
                    0x0e0147b79740d2e4,
                ])),
                Felt::new(BigInteger256([
                    0x96752d9793014c7c,
                    0xaf0636acd88a5586,
                    0xb576820f5d003397,
                    0x36352ce7afe31c2c,
                ])),
                Felt::new(BigInteger256([
                    0xc0c6143926b7ac36,
                    0xac6562fe62bec6d0,
                    0x26ff0209c8fd2da8,
                    0x2d3525863b647b13,
                ])),
                Felt::new(BigInteger256([
                    0x22fb6280a6b18a40,
                    0xa225017024468d65,
                    0x20b019babdfcf3a7,
                    0x2e8bb5b0d50db3f6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa5716638c3940c21,
                    0xa93b9ff17a2d5789,
                    0x60e295afebbe1486,
                    0x043245b7f435400e,
                ])),
                Felt::new(BigInteger256([
                    0xdd5b1dc6c98b8a7b,
                    0x6fec54ddf14a173a,
                    0xa5bf34632ae68628,
                    0x045c2593b2444b54,
                ])),
                Felt::new(BigInteger256([
                    0xaaf8d7f8eb2c0a2e,
                    0xb72a19a00ddd16d8,
                    0xc911e596952b8c33,
                    0x185d637c922f2cf2,
                ])),
                Felt::new(BigInteger256([
                    0x8382a80c82a7c9f5,
                    0xa99de751c2199bd8,
                    0x8a8e3c94ebc52668,
                    0x0c39fe773e84200d,
                ])),
                Felt::new(BigInteger256([
                    0x110dd265f1e6f903,
                    0xcfe63a83110dacc6,
                    0x64ae92f9b11ca820,
                    0x1a0c82ca93a347b6,
                ])),
                Felt::new(BigInteger256([
                    0xcaf5eb53e2db1028,
                    0x3935cf47583c882a,
                    0x3f7241d51e4c8cb6,
                    0x30ead19750745ca1,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x222b778014af1c09,
                    0xa011f4fa2536e3f6,
                    0x33ea663fa5980f20,
                    0x1f3f0ce61754bf2a,
                ])),
                Felt::new(BigInteger256([
                    0x8e10c842614502a3,
                    0x1a753c56ceb31d6a,
                    0xec3812bdede7891b,
                    0x1ecdcaf0842b810b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe460e150738145c5,
                    0xdf9d8a546067467f,
                    0x415c5284b09afa42,
                    0x03e03b3027c871b6,
                ])),
                Felt::new(BigInteger256([
                    0x65a4c53109967532,
                    0x4fc1b2f94ed581b4,
                    0x42efbce1b02f5b98,
                    0x0b5feb54a11846b2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf852b4015013a081,
                    0x0182dbc8a6bc8779,
                    0x04f2342cd486d8bd,
                    0x294e8ae8f771327a,
                ])),
                Felt::new(BigInteger256([
                    0x31377508b9145653,
                    0x23863d07b119d481,
                    0xc28d129b66e2959a,
                    0x37570df8521f4cfa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4e4452de15791924,
                    0x7a417c505f57c670,
                    0x131aba2c23d1de1f,
                    0x016ab90893403ab0,
                ])),
                Felt::new(BigInteger256([
                    0xac09159d9e2c969b,
                    0x59f7a4c001a12f64,
                    0xf590ce39aff3321d,
                    0x1e2292d66f6cb452,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x08a3a6b7f4d330df,
                    0x443cc73d3baa4e69,
                    0x5664605891e9947d,
                    0x0a64ca54288e4927,
                ])),
                Felt::new(BigInteger256([
                    0xd308d1dcae0adf08,
                    0xc8b89c5e188db3a8,
                    0x1e12b5909c6696a3,
                    0x139d0eba28a47e0a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0dfd9add924258c8,
                    0x1eefd0973ae3bfec,
                    0xb2e98170a884d708,
                    0x070fc8ee39588b0a,
                ])),
                Felt::new(BigInteger256([
                    0xb2d7a237100249de,
                    0x8554924943b17158,
                    0xd0b28e16b574eb08,
                    0x37c62a55bd267592,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc24719e26aa657a5,
                    0xb8fc293a135b7f65,
                    0xaf583c26687a7323,
                    0x0e2c98de87a3df38,
                ])),
                Felt::new(BigInteger256([
                    0x442bff2b0c165804,
                    0x07e6fc07b21797e3,
                    0xb6ae33c5cc32e4fd,
                    0x06fb14fb5401b285,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xad840b1df96a9752,
                    0xf391a0f06c3c7aa6,
                    0x48bb48fa6c6a9d51,
                    0x272d5e1ea669d671,
                ])),
                Felt::new(BigInteger256([
                    0x3fa2c9782e029ac0,
                    0x5e684e6a3bf94a95,
                    0x1ee64cba3bd23b3c,
                    0x28632bf89069e553,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbf1a8bd3a461bace,
                    0x9920667365f1b585,
                    0xdd942b6c24911815,
                    0x1cf510cfec7412b8,
                ])),
                Felt::new(BigInteger256([
                    0xa46106590b1bd729,
                    0xb0077323975dd4bf,
                    0xd00724622f2eb620,
                    0x077514f8a1fb99c5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7a9cc3fce724afe8,
                    0xa332b81adb8cc560,
                    0xb0518baf30418e03,
                    0x0dcc525b38d30cf9,
                ])),
                Felt::new(BigInteger256([
                    0xfc51ad9d7559b4d2,
                    0xd59a1147449c8178,
                    0x978adb8c704a6f65,
                    0x18cb1a942af82647,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
