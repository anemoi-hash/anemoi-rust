//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
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

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 32];
        for chunk in bytes.chunks(31) {
            if num_hashed + i < num_elements - 1 {
                buf[..31].copy_from_slice(chunk);
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
            state[i] += Felt::read(&buf[..]).unwrap();
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
                num_hashed += RATE_WIDTH;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        let sigma = if elems.len() % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        let mut i = 0;
        for &element in elems.iter() {
            state[i] += element;
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // 2*DIGEST_SIZE < RATE_SIZE so we can safely store
        // the digests into the rate registers at once
        state[0..DIGEST_SIZE].copy_from_slice(digests[0].as_elements());
        state[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(digests[0].as_elements());

        // Apply internal Anemoi permutation
        apply_permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiHash {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        assert!(k % 2 == 0);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = vec![Felt::zero(); STATE_WIDTH / k];
        let c = result.len();
        for (i, r) in result.iter_mut().enumerate() {
            for j in 0..k {
                *r += elems[i + c * j] + state[i + c * j];
            }
        }

        result
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
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x6d6e133fa988d506,
                0xae982e4f48b9b0fa,
                0x51be7557157b0ade,
                0x18e4648831e51f50,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x84f05c3c4c5a6758,
                    0x6990c80701f87b0d,
                    0xef44f2a808803937,
                    0x2f4669d8e6bc0c8e,
                ])),
                Felt::new(BigInteger256([
                    0xc906c2a115423eaf,
                    0xa34509050acdc9b2,
                    0xfed303789be3ffe0,
                    0x0b48f51975b8f424,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2a653baeb353f4cd,
                    0xbac35b136581a835,
                    0x5f2ba02e9e91ae1f,
                    0x2e3899ebc1909e3c,
                ])),
                Felt::new(BigInteger256([
                    0x67ba1b4825055d12,
                    0xb0d268cd6079bfc2,
                    0x7190cf4b03f87a29,
                    0x27d4624bf0e1a21e,
                ])),
                Felt::new(BigInteger256([
                    0x23cf3d033906942e,
                    0x365a45d500e3b051,
                    0xabb044d248611765,
                    0x25c27326c9d4bb0e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6edf59c1f1cc392d,
                    0x04f3a8350385e979,
                    0xa4989c79a3370068,
                    0x1f53b412e9458075,
                ])),
                Felt::new(BigInteger256([
                    0xc53061a50bcfbe6d,
                    0x3a68cbedcdf6ba61,
                    0x48c2b9d886313adc,
                    0x00acd5ebbc05822e,
                ])),
                Felt::new(BigInteger256([
                    0xd01ef7f6482d937c,
                    0x140fbfa30f080178,
                    0x68c3891c6eb50962,
                    0x1b9efbae1bc001c2,
                ])),
                Felt::new(BigInteger256([
                    0xfc87c3c7c1742e5b,
                    0xbd2c8d03f42e1b44,
                    0xc9fe9d2c143e7d36,
                    0x0c31f0ff9cdcbb44,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3c3b64d9a692030c,
                    0x6e92660d938f911c,
                    0xa2bd1db00abbe220,
                    0x094058d336bc7085,
                ])),
                Felt::new(BigInteger256([
                    0xf7ade9edeb19eb7a,
                    0x2f028119771742b4,
                    0x0cd17efb6f57d462,
                    0x20820e8764811153,
                ])),
                Felt::new(BigInteger256([
                    0xe896874fce9f2832,
                    0xbc944b9336ace4a8,
                    0x63bcb71d045b5b41,
                    0x0d244b4c4deaa2a0,
                ])),
                Felt::new(BigInteger256([
                    0x2d9c51521fcef56d,
                    0x612b1503ea51dc0d,
                    0xa4ae40b6da138a22,
                    0x02e95597a6432332,
                ])),
                Felt::new(BigInteger256([
                    0xa00b8016d0cb2ae3,
                    0xd5c571eb1f39b968,
                    0x0af1ad2e13e7cbfc,
                    0x17610cbc0f036561,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa68140907ae0c0be,
                    0x603d6eb7d0e97b2d,
                    0x519a440399716f93,
                    0x1d0f7b1c8034ce5b,
                ])),
                Felt::new(BigInteger256([
                    0x8c342a1ecc334401,
                    0xd17d87dddfb59623,
                    0x3eaf09c772fbc431,
                    0x2857ef104e88ccdc,
                ])),
                Felt::new(BigInteger256([
                    0x10eafc0afe2cc0cd,
                    0x4dfd876664ba6174,
                    0x5dc6a9ae9d86d815,
                    0x063994378aaf01e4,
                ])),
                Felt::new(BigInteger256([
                    0x0218130674382f35,
                    0x26fc725dd7f8b0b9,
                    0xd00f223179d01c1c,
                    0x22e73cc60d71378d,
                ])),
                Felt::new(BigInteger256([
                    0x5386c081f2c46b93,
                    0xc7131d4e93ab0b88,
                    0x3a37dd9c56c832e7,
                    0x0fdffb756e66d701,
                ])),
                Felt::new(BigInteger256([
                    0x0043228d239e4737,
                    0xbe8d675842b9785b,
                    0x9d99acb5829c95ac,
                    0x2df1ba52fd67feb7,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x871500363ff0e103,
                0x61724a57a843ab4e,
                0xb2a8a9d622f1dd37,
                0x1bf2e2dbd2275840,
            ]))],
            [Felt::new(BigInteger256([
                0x720dee4bba2642a9,
                0xf75c0a6681ee386f,
                0x64e933e8c92ade26,
                0x303b2d1dea51c9f8,
            ]))],
            [Felt::new(BigInteger256([
                0xeb1382fc618727bc,
                0x0ef6bd945c46aa52,
                0x4789fdbb5b891929,
                0x101570105f8b8d6c,
            ]))],
            [Felt::new(BigInteger256([
                0x496ed41e14ce2635,
                0xa6a51cec2753ff86,
                0x0e91308024baa184,
                0x1970a3adfc69a24f,
            ]))],
            [Felt::new(BigInteger256([
                0x92fc033ac1725c17,
                0x4692397b0e9f61da,
                0x499a6d78376ee3c2,
                0x18d5e9e865825122,
            ]))],
            [Felt::new(BigInteger256([
                0xda54f1da08f2a8e1,
                0x57c3b88feb802b36,
                0xc536b8651f5219c0,
                0x0e98a7080c2d7fdf,
            ]))],
            [Felt::new(BigInteger256([
                0x441fa01b4733546f,
                0x24b8a3b3c1c63cb2,
                0x47f6634182993931,
                0x2b1fc438ced3e2c1,
            ]))],
            [Felt::new(BigInteger256([
                0xf940f9ca26e7bb2f,
                0xf598d1da6cd4ea31,
                0x672bb3b9612d47cc,
                0x0f3a6419e7256ce3,
            ]))],
            [Felt::new(BigInteger256([
                0x4d8071016764633d,
                0x5dd186d4d424d993,
                0x7e0a765e68c681ed,
                0x0cf5a091cf6c66b1,
            ]))],
            [Felt::new(BigInteger256([
                0xf8c3eebef1aca6eb,
                0x02ede39dc3739a4e,
                0x5e756b2e062f3d2b,
                0x12b7fe38e74eb19f,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x05ab26a3e03dc27c,
                    0xa956f3372a21219d,
                    0x9977fcaad328eb03,
                    0x1db8dbde77eb45fe,
                ])),
                Felt::new(BigInteger256([
                    0x72589b5ff2517f55,
                    0x58e93b1ce132e857,
                    0xb2e4456b86361073,
                    0x178272d15aa1acbf,
                ])),
                Felt::new(BigInteger256([
                    0x7f473f05595459d7,
                    0x6c89c42764270937,
                    0xe4bf1a4a2d88a140,
                    0x26337f1998ddda04,
                ])),
                Felt::new(BigInteger256([
                    0x7423dc09c189ba4d,
                    0x9022f0277c76275a,
                    0x7c5fd3bcf6863d71,
                    0x28505e8071dc01a2,
                ])),
                Felt::new(BigInteger256([
                    0xd15d9045ac47a663,
                    0xd5e5b9aeca0675cb,
                    0xb352bef047e6c50d,
                    0x2c8c568b64f861ee,
                ])),
                Felt::new(BigInteger256([
                    0xd7109274c28361c7,
                    0x6d3f5eee8f9e7cfd,
                    0xe7a6dff26480411b,
                    0x1d0a8f72d6ad41d0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0a7c3673d8f075bd,
                    0x95739ef068bd6b25,
                    0xe99b58d8da384d7b,
                    0x18db332c9b53932d,
                ])),
                Felt::new(BigInteger256([
                    0x0cb79c15859f07c2,
                    0x2399ed9faae890ca,
                    0x757312abd74f149e,
                    0x1c2eff6b88703344,
                ])),
                Felt::new(BigInteger256([
                    0xa5cd805221ae58b9,
                    0xdb3498396de43d05,
                    0x06a7de273199249d,
                    0x29a6746fcb43ceb5,
                ])),
                Felt::new(BigInteger256([
                    0x79e89c83396066b8,
                    0x11eb3422ebf69ef3,
                    0xf37a628d87cb71fb,
                    0x229515bd9ad6cc70,
                ])),
                Felt::new(BigInteger256([
                    0xaab6678908337e97,
                    0x0a36d856794efacb,
                    0xee446e997c4898aa,
                    0x204b0b50422b79f2,
                ])),
                Felt::new(BigInteger256([
                    0x2dad59d400261fcb,
                    0xe8372f4ddfe362a1,
                    0x2180700093abc64a,
                    0x25c04c2d2dc9e7d8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2f5710452fecac3f,
                    0x2ac0aaa5ca9b3467,
                    0x1b02af85a78551db,
                    0x05e2becd2405e0c5,
                ])),
                Felt::new(BigInteger256([
                    0x0fa389a702bbc022,
                    0xf5e905329f064255,
                    0x846113c7064fcfcc,
                    0x125b9cb1ef644f5d,
                ])),
                Felt::new(BigInteger256([
                    0xfbd7dc8c51ef470d,
                    0x62f78f4f085f91d8,
                    0x3c4f8eacaa25367a,
                    0x2c15d085317c0c0c,
                ])),
                Felt::new(BigInteger256([
                    0x0fc9289d25f377c6,
                    0x5cf8847a4ec80177,
                    0x21a8879d7b887bca,
                    0x1fa9801f7788bbcf,
                ])),
                Felt::new(BigInteger256([
                    0xffcacf4a1b497fc5,
                    0xd4b23c2c266ec9ac,
                    0xfaebacaf80c0a76f,
                    0x06128c0afcbcc749,
                ])),
                Felt::new(BigInteger256([
                    0xf68a929be852ea90,
                    0xd1a43b5acc9ed1c3,
                    0xebfccc6fd40d590c,
                    0x1c61b8dbbd38f014,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7fe7b21bb8dbfc9e,
                    0x28d97dfa6809770b,
                    0xe361bbba20dfccda,
                    0x1861e0602b1890ec,
                ])),
                Felt::new(BigInteger256([
                    0x15dc59ed2d244c34,
                    0xcc7453ebad727dc2,
                    0xe3e3b6882b0ebfa6,
                    0x19f7cfce6d374756,
                ])),
                Felt::new(BigInteger256([
                    0xa29f092e8e96b77b,
                    0x7aedb9c1c618ae6d,
                    0x65c6b6556681568f,
                    0x2f6ec006880e3d4c,
                ])),
                Felt::new(BigInteger256([
                    0xb89ddc297bce84ff,
                    0xa10f37a7c4664a59,
                    0xae2c5bc8b66d57ea,
                    0x2bec4b1e3584a6c8,
                ])),
                Felt::new(BigInteger256([
                    0xa48fa166cfac7655,
                    0x7a5ca8915f3d7ccc,
                    0xf6ed1094704fcc62,
                    0x04bb309a7ca68887,
                ])),
                Felt::new(BigInteger256([
                    0xbb561c9a67c1e401,
                    0xc670ee2ee5a0cb36,
                    0xb7ad6f0183e799e0,
                    0x25d5576de75c42a4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd5a21ef38bb61e82,
                    0x840989628046145a,
                    0x19bf1b870a04bc36,
                    0x0cff22a174f803f3,
                ])),
                Felt::new(BigInteger256([
                    0xf705fa89a0cc56f0,
                    0x0389fd7e21083139,
                    0x4a6c5f6079db5aa5,
                    0x0a7a24adb1043740,
                ])),
                Felt::new(BigInteger256([
                    0xa55b5ff71e1bce91,
                    0xd33e5a561583f593,
                    0x45c59ba8cdd1750d,
                    0x26ef829d96474f6d,
                ])),
                Felt::new(BigInteger256([
                    0x2cf5da958057b9b1,
                    0xb13127440c427674,
                    0x062cb2d849603603,
                    0x11b640ceedcfe74e,
                ])),
                Felt::new(BigInteger256([
                    0x631000b1e9848c77,
                    0x33565c7f6ee8048d,
                    0xb54bb1fb99649b9e,
                    0x10344c8c5290bd86,
                ])),
                Felt::new(BigInteger256([
                    0xfc37a4fa1bbe7cbc,
                    0x21d2017f4a093f32,
                    0x25f3452a7b083c50,
                    0x099ce2cb8c83e6b3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7489b080234b7811,
                    0xfdee6195faa20fae,
                    0x3af6ccc4b78251e6,
                    0x0187e74a4e5250a3,
                ])),
                Felt::new(BigInteger256([
                    0x10de436df6a7c6ea,
                    0xd4c89236a0888680,
                    0x2d0146b1a868f091,
                    0x2ee4f10f1e8ce665,
                ])),
                Felt::new(BigInteger256([
                    0xad493dc643e96b56,
                    0x91138ff18f75c18e,
                    0xebe0903ad0e9a440,
                    0x0bea35a55a013ca6,
                ])),
                Felt::new(BigInteger256([
                    0xe9e7417c5d927f3e,
                    0x2a66e5b4af6084f3,
                    0xc8cef0b28890edd4,
                    0x055db67a231d5dc4,
                ])),
                Felt::new(BigInteger256([
                    0x7b4f241e3ce171ea,
                    0xa1141c9685014ac9,
                    0xf56436b81e8c36bd,
                    0x09316605f9037767,
                ])),
                Felt::new(BigInteger256([
                    0xe446f387be0de04d,
                    0xa05ab89024a83f6c,
                    0xe5f286dc97e469ad,
                    0x007b957a297220c3,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x6e31cabb7524781a,
                    0x02c2553fdadff1ee,
                    0x06c52127c2ace7e4,
                    0x1c1662d8bd856c3e,
                ])),
                Felt::new(BigInteger256([
                    0xff678d4c7aeb9e30,
                    0x92a7a033b98b3003,
                    0x7d273e66b2d32d17,
                    0x123b1431af6c6f89,
                ])),
                Felt::new(BigInteger256([
                    0xca3b7aca31af605c,
                    0x8dcd4671f449bb56,
                    0xec3c790bf0ca1aee,
                    0x2f84383e0f6128e4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1c844a8b94eba389,
                    0x5c51b012b48ebbdd,
                    0x8dd1f8473e75d399,
                    0x2af0000d30876283,
                ])),
                Felt::new(BigInteger256([
                    0x5a41d016d49964f5,
                    0x43cdd2aed92448cf,
                    0x92eff8e7944f7c7c,
                    0x1d219720632c5a0a,
                ])),
                Felt::new(BigInteger256([
                    0xf096041441abf599,
                    0x850ae48bf3d56fe9,
                    0xe3f1b98b5557b504,
                    0x1ccae1e77ccf892d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb9c28fc81d6dde85,
                    0x547ffc43ca5a7a95,
                    0xf0f7a2f87669c42a,
                    0x24497c8c8ade29fc,
                ])),
                Felt::new(BigInteger256([
                    0xe1bf44a6ed0835cc,
                    0x7186031ef1ff20c4,
                    0x42e17bd4745a40c6,
                    0x1418a831b01df95b,
                ])),
                Felt::new(BigInteger256([
                    0xf9a38686a369ee24,
                    0x50d1dc53273d75b2,
                    0x4b217bed7b3bd8a1,
                    0x2721f1469e2e2b0f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3e1a2180df9ab299,
                    0xdb1e44e22e48e22e,
                    0x0194026ebce7840e,
                    0x15a501eeacf8542d,
                ])),
                Felt::new(BigInteger256([
                    0xbb88b3d6db8fc2df,
                    0x05ec06e7afc84c37,
                    0xe3b2d5a6c08d95eb,
                    0x0d81dc7053013d65,
                ])),
                Felt::new(BigInteger256([
                    0x3958a9a1d69ac48c,
                    0xacfa0d7200ee2564,
                    0x9d6a47d6a028de9d,
                    0x1358c561ae2e3767,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x38006ca2207766f3,
                    0x888315ed1c55e909,
                    0xf8ef4226c8a3d3c2,
                    0x2122bc7d57ed99c8,
                ])),
                Felt::new(BigInteger256([
                    0xc7aa128fe3508fff,
                    0x93f4b7bda281efce,
                    0x5cf95d524dca75d6,
                    0x219c644087ed52bf,
                ])),
                Felt::new(BigInteger256([
                    0xeab3aa0e92b93e66,
                    0xef54148ae92f09fa,
                    0x42956a2df2a703d0,
                    0x132245c67e708eef,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9e743354e9f5ac34,
                    0x2629e995a8ad0cac,
                    0xc8fa4171baa99dd1,
                    0x032c2dd4735c9778,
                ])),
                Felt::new(BigInteger256([
                    0x2157f3b10e846a81,
                    0xdab47d88889c404a,
                    0xc3aeee9bc8ea728f,
                    0x03f83a456ce39977,
                ])),
                Felt::new(BigInteger256([
                    0xeadff0f8d2ad9f5d,
                    0x35e4a0e51ea02437,
                    0x0be391c4a356e41d,
                    0x01c04a8b32d2de07,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2d23bdee35e6f778,
                    0x44b98623dbaa477c,
                    0xfce00b898a739b8f,
                    0x2628f6665e179e5a,
                ])),
                Felt::new(BigInteger256([
                    0x1618ddab05708b8e,
                    0xb4b6a1be438d4bbf,
                    0x040c5af19ebeff87,
                    0x23fbd3f6488753e1,
                ])),
                Felt::new(BigInteger256([
                    0x694e0dda22fe4e57,
                    0x1f0ca988abfc5721,
                    0x3276d48a13f237b6,
                    0x1baa715aa6fa36eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x305ba40859bc8316,
                    0x7fb769151a24256e,
                    0x81d244379635d86c,
                    0x02cbd2d0e8ee9cf2,
                ])),
                Felt::new(BigInteger256([
                    0x04750210eadd7bc4,
                    0x3a2cfb30d89afc20,
                    0xca639ef4a009c6cc,
                    0x2ed7e648277c1cd0,
                ])),
                Felt::new(BigInteger256([
                    0x29b05ee5fd2a5dd7,
                    0xaca177cc85630245,
                    0xa4b6890eaf2ad866,
                    0x1b19f6ddeca6c0f6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x04eb4c2eac9730bf,
                    0xe452fa17b87bdfc4,
                    0x92ea1548ffac8e3e,
                    0x0f9c9638bc640ded,
                ])),
                Felt::new(BigInteger256([
                    0x8ff876ef6c70aad5,
                    0x1a3a12eaa30241b4,
                    0xe507eba8ef7837f9,
                    0x03ce61d2005802c2,
                ])),
                Felt::new(BigInteger256([
                    0xab3276855dced8e9,
                    0xe7e70660524eae40,
                    0x4d3a4792a075d634,
                    0x1968d81e5b0c31f3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4c0cc1ed09d0bafe,
                    0x531c476de5760d35,
                    0xdc25f18dbe9839ea,
                    0x13151ee0e6a376b3,
                ])),
                Felt::new(BigInteger256([
                    0xfd3325543107994d,
                    0x48a16bb03da0d46f,
                    0x9584e225eb2aca33,
                    0x2206b1f33cca539b,
                ])),
                Felt::new(BigInteger256([
                    0xa67c6d54090ecee5,
                    0x8a4076b80a83a2d0,
                    0x29fbe94d60905820,
                    0x29384d2c76c7a730,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd3a00a69c8a5d50c,
                    0x77205ee3056aab18,
                    0x09d3c8fad75914e4,
                    0x1c5150983980b3e6,
                ])),
                Felt::new(BigInteger256([
                    0xdbe51904135d286f,
                    0xa23981aeebbc8781,
                    0xb0258b1430d4e38b,
                    0x2dd6f8b5c2d529c8,
                ])),
                Felt::new(BigInteger256([
                    0x836fc582e4b881d5,
                    0xa8519ccde211ed7d,
                    0xdb8e5b65cbd131c2,
                    0x12dfc959680d7c49,
                ])),
                Felt::new(BigInteger256([
                    0x1334745b15c89684,
                    0x7c91ed1f2fcab7f1,
                    0x2595111c9283ee51,
                    0x1818e51643991602,
                ])),
                Felt::new(BigInteger256([
                    0xd0475cd3b93fe9b7,
                    0xadd6a6bc99a4772d,
                    0x3b601f157d43fcb7,
                    0x189972470fbe3bf3,
                ])),
                Felt::new(BigInteger256([
                    0xf5020f2902fcd2f0,
                    0x34fe57b29b537c2c,
                    0x6accd51cca1a2803,
                    0x02726e3b506d8de2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5670dce54701e65a,
                    0x7c07c7668a856cc2,
                    0xa3dc1d2223a54261,
                    0x2f3eeb4b3ab664ef,
                ])),
                Felt::new(BigInteger256([
                    0x2dd2b6aec0e5565b,
                    0x9adb1c33ffa9a2de,
                    0xc02c07840c79b941,
                    0x2a3efbb6da142848,
                ])),
                Felt::new(BigInteger256([
                    0x0c48db2b9f1a14ae,
                    0x0e26426dfff26fef,
                    0xd3900ca9f803b716,
                    0x0f2a0c8f53c127f9,
                ])),
                Felt::new(BigInteger256([
                    0x436247b836ad4cde,
                    0xa144165c46ac2ddb,
                    0xeebc8a690af982f3,
                    0x01eadf8a5930bdba,
                ])),
                Felt::new(BigInteger256([
                    0x93c86bbcfaafeb9f,
                    0x0a8c67fce202b767,
                    0x0b529ef05d97a2dc,
                    0x11b87fb601d71616,
                ])),
                Felt::new(BigInteger256([
                    0xec13fe23c67cc40e,
                    0x8e7dadbef7bd35b6,
                    0xbf60ca7aa26c773c,
                    0x26139e14de49f703,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd170bf01835a6ff2,
                    0xfe36bd3f527d8096,
                    0x4640b5f2d287b5c8,
                    0x0b8ab47d52b26468,
                ])),
                Felt::new(BigInteger256([
                    0x17703b4865f29180,
                    0x1a8b06c807caaabb,
                    0x38479edc0961fa06,
                    0x001746e7555fc49f,
                ])),
                Felt::new(BigInteger256([
                    0xe0c8b1e1bbba307f,
                    0xcff624d10c505337,
                    0xd1f491da93877ffa,
                    0x2d58fd7dc8403c24,
                ])),
                Felt::new(BigInteger256([
                    0x84888573dd2d91dd,
                    0xdfa560786975cdf2,
                    0x08bde6efb108ac28,
                    0x2f845a0c92600ff1,
                ])),
                Felt::new(BigInteger256([
                    0xd83ee9b5d1dbd042,
                    0x3e8d57746ab0e122,
                    0x602b0dc32e9d6a83,
                    0x2121fdd38e0658fd,
                ])),
                Felt::new(BigInteger256([
                    0x36c840e603c9f620,
                    0x0e0ea5986c8936ab,
                    0x9c73b883c9cc0f5b,
                    0x1200a8ad6a79f0ac,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x20ffa1ec30eb156e,
                    0x12745d95a56018d3,
                    0x958dd62ca6d562ef,
                    0x27460c7cc42138f8,
                ])),
                Felt::new(BigInteger256([
                    0x1ddacdb5a19cef9d,
                    0xd225c0d11c33fc75,
                    0xe1f2df12db261702,
                    0x0efcd98f90857382,
                ])),
                Felt::new(BigInteger256([
                    0xcc9cc69b8cf09fa8,
                    0x5dbf5148827c0595,
                    0x93b804924ed32891,
                    0x121ccb67e15313e5,
                ])),
                Felt::new(BigInteger256([
                    0x7f1fe5c7f83656ab,
                    0xb552f565acbb29f5,
                    0xc7ed392603802745,
                    0x1e42f2eda7f1377d,
                ])),
                Felt::new(BigInteger256([
                    0xd26386783327b566,
                    0x5be186eeda6b2d4a,
                    0xef85ebef84ceedad,
                    0x1865e51f2f24ba08,
                ])),
                Felt::new(BigInteger256([
                    0xa0d13955a0659131,
                    0x164949192df3bb30,
                    0xc40b23e5cfded2f9,
                    0x0e58161e3f1e1d39,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcde7faa9b7bb937a,
                    0x9cf45198b3170cd2,
                    0xddac5bc3afa5e7de,
                    0x1901704a8ff680f6,
                ])),
                Felt::new(BigInteger256([
                    0x4fa06c4da895a450,
                    0x44b6c3fd2b375dec,
                    0x7f1cc3d2bbd8c799,
                    0x17f0b370a2385460,
                ])),
                Felt::new(BigInteger256([
                    0x9c14801c52494c8c,
                    0xcac32c9f12d8628f,
                    0x7a7113d18f1a19a3,
                    0x11a98289f032b0d2,
                ])),
                Felt::new(BigInteger256([
                    0xba29bb0382ec15b9,
                    0x7a0da38c9eb404bc,
                    0x8071934696dcd25e,
                    0x19738002a9836ed4,
                ])),
                Felt::new(BigInteger256([
                    0x4c2f2a7d81dc95fa,
                    0x293921bd7c0c80bf,
                    0xf0bef7c64cb8d37d,
                    0x2b58af759faf967a,
                ])),
                Felt::new(BigInteger256([
                    0x7e3b8d8f80321c75,
                    0x29964e2a6c24ed77,
                    0x887c550631e19697,
                    0x28a261eef5190337,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa5286cd442c6fa32,
                    0xd1a2278418857e97,
                    0x960c4f67ffef69b2,
                    0x12ba573cdbff05fc,
                ])),
                Felt::new(BigInteger256([
                    0x3dbfbab5e85e25a7,
                    0xd13b1b2df80b082e,
                    0x290fcd35e64a2b43,
                    0x0063bb19163bd5e8,
                ])),
                Felt::new(BigInteger256([
                    0xefc46417cbc0de52,
                    0xa64fbed9c59a7fa9,
                    0x8e27189358f66146,
                    0x290dbd157b3721c4,
                ])),
                Felt::new(BigInteger256([
                    0x6493b376d0aaa2b7,
                    0x171cc501f802f911,
                    0x58e1d094ca979d76,
                    0x04e96b7d3146e3d6,
                ])),
                Felt::new(BigInteger256([
                    0x40c5046d603ed3ec,
                    0x3e0703b6f28f106c,
                    0x4e997df4ad857c58,
                    0x29917886b809ec68,
                ])),
                Felt::new(BigInteger256([
                    0x857a1cac3c52103e,
                    0xc7341c10f8edbf4a,
                    0x95c2ac4c0c667386,
                    0x2c6031caf5f2266c,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xfbb446bb4942795f,
                0x8bb5d154204312bb,
                0xb7d892e3e4c8d78c,
                0x2d7160d59b216482,
            ]))],
            [Felt::new(BigInteger256([
                0xef1b0688fa370389,
                0xf627922ab0a4df7b,
                0x94131f4d251a545e,
                0x0413dc2f4e200568,
            ]))],
            [Felt::new(BigInteger256([
                0x5904ceded563052e,
                0x7f5671247b254680,
                0xc6aa5503e47e8534,
                0x2f1fc791f7f8ae3d,
            ]))],
            [Felt::new(BigInteger256([
                0xf6daf2e2b9483cbd,
                0xf682eeaa768d893c,
                0xca60da359c1ca039,
                0x061b554dccf628d0,
            ]))],
            [Felt::new(BigInteger256([
                0xe5e0972bf99bf519,
                0xe39b8731fb692219,
                0xd8fa5acf0ce6b659,
                0x0c75de175220fa9c,
            ]))],
            [Felt::new(BigInteger256([
                0x876c4d0746efcf6f,
                0xe7e23abc6cf4af40,
                0xa85fb3fb6186b71e,
                0x2d298765652982dc,
            ]))],
            [Felt::new(BigInteger256([
                0x4e90cd59e985168a,
                0x2f65acb79c633425,
                0xb60dad775c5ad603,
                0x19149cdcf786690b,
            ]))],
            [Felt::new(BigInteger256([
                0xde1e1a27c2f2903c,
                0x81f3da548a1f90e9,
                0xa29ac311475b1001,
                0x297a2a1bb68531b6,
            ]))],
            [Felt::new(BigInteger256([
                0x231ce1021832f99d,
                0xc32fc4d5dd354b3b,
                0xbf629499f8b222a3,
                0x22b4f340b8e75716,
            ]))],
            [Felt::new(BigInteger256([
                0x31d29e7091f5529d,
                0x487787b7cbb35ca4,
                0xb17f3650da6dc249,
                0x1198877c5396e02c,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
