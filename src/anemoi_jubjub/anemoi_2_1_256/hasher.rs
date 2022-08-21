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
                0x22a2391cc44722c1,
                0x6a81b0806c357346,
                0x70db45a1e110c3e9,
                0x0eefb713b7cc7a4e,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xd45a583cd2ce4829,
                    0x42fe1504cf39c5a4,
                    0x814883257133ba55,
                    0x04bd116ee5268e24,
                ])),
                Felt::new(BigInteger256([
                    0x91210c5663ef1faa,
                    0x17f822a88477cb65,
                    0x263d9137340cd108,
                    0x01af02cf70844fe0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdef8eb239acb872c,
                    0x470046178792bb49,
                    0xd1424db8aabade25,
                    0x5219b68f9254d44c,
                ])),
                Felt::new(BigInteger256([
                    0x4bc0a091b5d28b63,
                    0xf704531378b4ca5a,
                    0xa787bc2935ccabb2,
                    0x4fc2c24a31a71874,
                ])),
                Felt::new(BigInteger256([
                    0x05f5d28e812a2cb4,
                    0xeceb8b69b0c5c92e,
                    0x8ac806ad17e1b249,
                    0x38c62c9568f7b2bd,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3dfced05dbf897e7,
                    0x5b78632dc884f9cf,
                    0x0ae2cd11efed5d36,
                    0x198b571d29ef80ea,
                ])),
                Felt::new(BigInteger256([
                    0x7e834fa785e00d26,
                    0xbac4f085f497746f,
                    0x2e3cc32c8bbd3a41,
                    0x5c1eb7a8c6e3e005,
                ])),
                Felt::new(BigInteger256([
                    0x92caf79fd6afa6b1,
                    0x6ba897ef4946328f,
                    0x5cc0fdbdec3e6473,
                    0x076b9de7696a5664,
                ])),
                Felt::new(BigInteger256([
                    0xda7bbc9d00b15c33,
                    0x35bf20e581f2ea47,
                    0x0d04ecacc2d4d97a,
                    0x5886900cbbc2cf5c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe258ac53d5455aa1,
                    0xd352cfe92f843ee0,
                    0xe6af2266335e537d,
                    0x5cd820751d9000b0,
                ])),
                Felt::new(BigInteger256([
                    0x49194b6b9dc783c5,
                    0xa7f341912466a841,
                    0xb7fa89a00e40a502,
                    0x699f73e4c7e3af77,
                ])),
                Felt::new(BigInteger256([
                    0x0e1cdc86e040d9d3,
                    0x8a7754626366bfae,
                    0xab892b73efe153a1,
                    0x04629a1f655470cc,
                ])),
                Felt::new(BigInteger256([
                    0x2525804581c071ff,
                    0xc262e623a2a2375a,
                    0x8122cc5180d617ad,
                    0x2e7ca4a1611fd97d,
                ])),
                Felt::new(BigInteger256([
                    0x6cb8402aaf0ca038,
                    0xbd3004b50b09b9bf,
                    0x603b9b5490b06a8e,
                    0x0f778fdd0f1bf5ff,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x43cbc3c4e27f58f2,
                    0xfee70ae1cab18295,
                    0x040b8c0e1816f548,
                    0x304d9b8bbe99082c,
                ])),
                Felt::new(BigInteger256([
                    0x87e8c4a0ac181c4b,
                    0x339c58a7c437b24c,
                    0xcfec95106de9cba4,
                    0x42199dd3aef082bb,
                ])),
                Felt::new(BigInteger256([
                    0x94641bc2c34f77fe,
                    0x3628f3d8e8943ce1,
                    0x5a00c2baecba9c31,
                    0x411d82e0958218e3,
                ])),
                Felt::new(BigInteger256([
                    0xd066e3254d1596d9,
                    0xccbe5ff10a91586a,
                    0xe6a0cac3edc980d5,
                    0x3a9bec9da8b3ce00,
                ])),
                Felt::new(BigInteger256([
                    0xc3781ed08d0c78ad,
                    0x1bdd9c5c5ced7455,
                    0xddcfd0519583367a,
                    0x6195ddffd2474bea,
                ])),
                Felt::new(BigInteger256([
                    0xd63e2049f13a14e1,
                    0xcf33fabeafdb7836,
                    0x46c10e46b9c4798b,
                    0x3b611ebe16e99791,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xf9e45340012e5c79,
                    0xa169383a1701f704,
                    0x2c94518443140a8f,
                    0x47101c7cc67d0e73,
                ])),
                Felt::new(BigInteger256([
                    0xeb09a969c7f9ca7f,
                    0x0071da9461f66501,
                    0x040bae03c75451e6,
                    0x47b24c7d6227094a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0560a39335500c58,
                    0x090bf4639cf728f4,
                    0xb5083b4a0359d909,
                    0x540495039b86ed3b,
                ])),
                Felt::new(BigInteger256([
                    0x79b27127b73e878e,
                    0x35b7821af4fa6902,
                    0xa08561149f388686,
                    0x68a57236102532b1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5870042e8919c316,
                    0x828f873197218b12,
                    0x6f88c127d1d7786d,
                    0x1b7a754006246cf8,
                ])),
                Felt::new(BigInteger256([
                    0xcd8fb257ee1e30ef,
                    0x5c83824e243afe21,
                    0x34d348656f5d1977,
                    0x5b5a7de248a4e8ad,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xba7c04c954cdf30d,
                    0x67a416bd80d09aac,
                    0x99b6b3ce653a56e9,
                    0x3a77791172c9f20b,
                ])),
                Felt::new(BigInteger256([
                    0xaf92b3f729823b2c,
                    0x616ac7eff6dac01f,
                    0x1b05c17fb6f12254,
                    0x16220e79e9cb63ee,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb7788b11037e795c,
                    0xc37f23486004fc98,
                    0xea6e97c26a7c5951,
                    0x599a27089d1078c7,
                ])),
                Felt::new(BigInteger256([
                    0x8c098c9212455289,
                    0x82b2467174af02d0,
                    0xf5c670654949885e,
                    0x1e1ae17705bf40ce,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xab409a4d282b5e8e,
                    0xc7647345f2a347b7,
                    0x5bb6f1bafd3066a2,
                    0x1850a0e28fe76321,
                ])),
                Felt::new(BigInteger256([
                    0xe97f9795ec03ecaa,
                    0x020b5cac7d5f5f53,
                    0x2810670fec81b069,
                    0x418418ef5d8f7d20,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04dfff0dd57952d9,
                    0x7f2d6edfe101a7c1,
                    0x366ffd3a3dfde692,
                    0x04036d6e745650c9,
                ])),
                Felt::new(BigInteger256([
                    0x98599367983b4fd9,
                    0x0e324c33ba56745e,
                    0x243ea157ca7f9ce7,
                    0x0d5f31d17d18975c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8a3f9b9bf84f8f2e,
                    0xf270c1221a3b1a79,
                    0xfcdec3a04e6e7bf8,
                    0x60f11e3870b70791,
                ])),
                Felt::new(BigInteger256([
                    0x1a13bf7020372c10,
                    0x1d798e34b5e68510,
                    0x05b01dd53c43af17,
                    0x22e1e7603f8fed40,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb571b0959ea1586e,
                    0x81a6f26601da28a6,
                    0x1b11c503eea5ca8f,
                    0x614503cac354b27b,
                ])),
                Felt::new(BigInteger256([
                    0x4c8be988e421d853,
                    0xdd90f9150de45ce4,
                    0x4a3633780b61a044,
                    0x1f335da43f7375fc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa91364891d2b6e2e,
                    0x3dd933f8dbd60e38,
                    0xb60902dde3e8725a,
                    0x02b1f6247510c459,
                ])),
                Felt::new(BigInteger256([
                    0xc8d9d0ac7c14b485,
                    0x23f3aa1013a6bde6,
                    0x78153810c6acdeda,
                    0x6ec6d6970b4e6038,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
