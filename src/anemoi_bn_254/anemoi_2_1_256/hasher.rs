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

        AnemoiDigest::new(digest_array)
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

        AnemoiDigest::new(digest_array)
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
                0xfc0e0362b4affe8f,
                0xd2d0f82b7d0313ac,
                0xd518acd81b882c64,
                0x0ed49ee7f7faf835,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x5c51a00c151ebe5a,
                    0x089b329da1203060,
                    0x508270a67fc0ddab,
                    0x1951e9cc3a059e7a,
                ])),
                Felt::new(BigInteger256([
                    0x4ee2dc77c2b7e163,
                    0xb20f480df182156b,
                    0x5a1c0f4bf17a2047,
                    0x1105aba03cacb86d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x28edb4fe4708490d,
                    0xfe892a3fab186367,
                    0x06fa7b7d6dec039f,
                    0x04f76d53c4d16d0d,
                ])),
                Felt::new(BigInteger256([
                    0x5274f01dd65befed,
                    0x77c88efe5068de80,
                    0x0e0fdbc4bd836b42,
                    0x28b01c5f4a030a46,
                ])),
                Felt::new(BigInteger256([
                    0xf4c97e9f7246b9a1,
                    0x2976ddcb382fba47,
                    0xecd6bd5fd021c18f,
                    0x278583f0823fae88,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb9a873de0446d1c5,
                    0x09739484305203e3,
                    0x0bfb482481bb084c,
                    0x2288d6fbba1af7ad,
                ])),
                Felt::new(BigInteger256([
                    0x62e9fad6ea924988,
                    0xd8554bd902e5cf4f,
                    0x37aa58972d355716,
                    0x05a9da08979b0675,
                ])),
                Felt::new(BigInteger256([
                    0x9b92e739c875a4f3,
                    0xb7b969d257573af8,
                    0x93964bee2e1bddd5,
                    0x29ac4ce3abd8a8a6,
                ])),
                Felt::new(BigInteger256([
                    0xcbc68170f49e5314,
                    0x1bdf0bd9ec00546a,
                    0xaf78b1c4925e7e1d,
                    0x2cdd12d3b5d23fd3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8bd790783dc06e28,
                    0x02a81d1e6aec0679,
                    0x23ef0a937ceb5cb3,
                    0x1542560b3f07a52a,
                ])),
                Felt::new(BigInteger256([
                    0x9cc8a9f967996bda,
                    0xcd0257e82adb03ec,
                    0x7380b93a5a19efad,
                    0x07ad6093835ad770,
                ])),
                Felt::new(BigInteger256([
                    0xf52cac550c118865,
                    0xfbe7ea5007cb0296,
                    0xf95e34e337fe7520,
                    0x170c8048fd2ffdee,
                ])),
                Felt::new(BigInteger256([
                    0x605dca203497b888,
                    0xe74e472d9d300a0a,
                    0x2d33b4aa25a00bd2,
                    0x08a3fed8c1c452a6,
                ])),
                Felt::new(BigInteger256([
                    0xb861d64cb62f0ac1,
                    0x1b6fa04dca3dbc7f,
                    0x07ba006bc9fbc376,
                    0x2a8b1fb202ed69b5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x66eadb2c962a2440,
                    0xbc89f171fcbcc04a,
                    0xd494ad2682ac0db8,
                    0x2c1aa4817c9d7305,
                ])),
                Felt::new(BigInteger256([
                    0x1008e7178592bc1d,
                    0xc0cf1cd883f10783,
                    0x480b4847689c42c7,
                    0x0e803247891b0e2d,
                ])),
                Felt::new(BigInteger256([
                    0x8b6ff37683846d42,
                    0xe9903be6f73baede,
                    0x9085d9563aa1c1b2,
                    0x01fec27f23de0c66,
                ])),
                Felt::new(BigInteger256([
                    0x267d35bdd1ef66a1,
                    0xf1edaae15b278dd6,
                    0x2b111b1abf8021d7,
                    0x1bf07bb950e94513,
                ])),
                Felt::new(BigInteger256([
                    0x1f1e5152c30619f9,
                    0x80596376e2104109,
                    0xad389b2f9036d2b7,
                    0x062a3b5a1aca6e97,
                ])),
                Felt::new(BigInteger256([
                    0x24c35fc8620242d9,
                    0xd4ea80ddca74f6d6,
                    0x2ef51735a4f3f2b6,
                    0x23c962406e1a58f7,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xaa00600f1ced6ba3,
                    0x940259692f2d51db,
                    0xbd7cf988ddf01503,
                    0x1d30e7e02715be54,
                ])),
                Felt::new(BigInteger256([
                    0xfe6178787f5f60da,
                    0x6cb721b19c9c3d28,
                    0x35176577dd53b8c8,
                    0x140c896cd0fecd63,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9bc42488fa3967e9,
                    0xc2bb9e6b23467743,
                    0x51abdc87cdb16ff8,
                    0x070277df1c70a056,
                ])),
                Felt::new(BigInteger256([
                    0x1afb5e9275fa2852,
                    0xe530e779cf509079,
                    0x50666d77f5672769,
                    0x247071d33e495731,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xca9c4f734380a00d,
                    0x157f7eaff7b47798,
                    0x6aa83754a248830a,
                    0x0aa8fd8e1a83488a,
                ])),
                Felt::new(BigInteger256([
                    0x800490c7ba005a8b,
                    0x22b228bcebe86173,
                    0xe04119803749a035,
                    0x175c29d22e6f87e4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe965633c65ac71d7,
                    0x8fae52092c41a190,
                    0x116af6b9e7fa207e,
                    0x245cac2054f8b2d6,
                ])),
                Felt::new(BigInteger256([
                    0x4ebfb6cd8a7e6314,
                    0xd345f5a489cbd3de,
                    0x55eee44dbd68fc6b,
                    0x03f2418f45630e1c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x177a74af5b269ee7,
                    0xe57cd2bbf570d65f,
                    0x162e1b7122a68d49,
                    0x0cd04166c4a90b94,
                ])),
                Felt::new(BigInteger256([
                    0x7215cb96ab436d61,
                    0xfedb94ac870522fc,
                    0x7a5011d58ce5d7e7,
                    0x27299e64f32851d3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc9f03f1e0e08f77a,
                    0x35032fdbce0c703f,
                    0x32b3fb4ee5bdfd4f,
                    0x03f10bfb2ac908c3,
                ])),
                Felt::new(BigInteger256([
                    0x955d32429712e1dc,
                    0x795fd47129dec1b7,
                    0x92c87acb393c5c77,
                    0x12af7e4e684d2090,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb112492a7eb00b42,
                    0x719420d1f2fe00df,
                    0xd277bd05739492dc,
                    0x00eb3cf830bc6f53,
                ])),
                Felt::new(BigInteger256([
                    0x49f38cf77c3ff05d,
                    0x0fc999751aeef056,
                    0x784622b219a57889,
                    0x152315d1bb7f976d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdfdf132cbef2a0ea,
                    0x1131c900387fb12c,
                    0x286b2d521fa8075f,
                    0x138b59220273594e,
                ])),
                Felt::new(BigInteger256([
                    0xa7750f7756506986,
                    0x02399d8c468174f9,
                    0x34a274521fd07ea0,
                    0x2e9a2f980e596eeb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xc1a5d5c3fd7a4d98,
                    0xc0b102a0c4f98c5b,
                    0xc7e1204726749ae7,
                    0x19a203a34c2e8e1d,
                ])),
                Felt::new(BigInteger256([
                    0x724e1a3945a921e8,
                    0xaa244621daad4a10,
                    0xade265208ce3fc2d,
                    0x189721419d7963be,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9388b72e69c52db2,
                    0x1b5835ca4d33705d,
                    0xad7c8ce2509275c5,
                    0x2a2aeb15c7644632,
                ])),
                Felt::new(BigInteger256([
                    0xaa969053e321c26f,
                    0x2b4f1d07dc424a3d,
                    0x3c70b632b454e95b,
                    0x21e29a772b1c9108,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
