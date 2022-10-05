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
        // We can output as few as 1 element while
        // maintaining the targeted security level.
        assert!(k <= STATE_WIDTH);

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
    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x1b3eedad434c9ee3,
                0x995656a9a385d298,
                0xd5a347123b4e01cf,
                0x232aecc3fce13f2d,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x2c45bc4b06488def,
                    0x6dd6834cab708553,
                    0xa5709263e53fdf4f,
                    0x2a37eb447a7e2b24,
                ])),
                Felt::new(BigInteger256([
                    0x114e1a78068dac63,
                    0x16b2baea5a55a417,
                    0xb6971a6a61aff10a,
                    0x2ea59e0190675b92,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcdf1e79de6edcd8f,
                    0xbb81b49f6e5d2430,
                    0xd882c005c49185ad,
                    0x246d9911b4a91411,
                ])),
                Felt::new(BigInteger256([
                    0x5de7fdc0eab2faf6,
                    0x335805b3f7de239e,
                    0xe81bd240c8b66d21,
                    0x30326eba6ae2f86f,
                ])),
                Felt::new(BigInteger256([
                    0x18024198c5862181,
                    0x74516fccfb9563be,
                    0xb99a15f07b645db0,
                    0x023645e94d1cebf9,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf52379df26240300,
                    0x1b2e1f200e93c0aa,
                    0x4d038595592b569f,
                    0x036e242a3489420d,
                ])),
                Felt::new(BigInteger256([
                    0xe51c2f031e36b551,
                    0x58ac8e786081501d,
                    0xafa9f8c90d4b74cb,
                    0x2816e2a8fe5fd9ad,
                ])),
                Felt::new(BigInteger256([
                    0xa3f29c9bea4106bb,
                    0x36163d3ceb8acefd,
                    0xed708b54c9d16ac3,
                    0x0f35dd03411ec6b9,
                ])),
                Felt::new(BigInteger256([
                    0xc601f1435f2d3953,
                    0x1cbae9205e18b8a0,
                    0xfaf17b835a88963a,
                    0x15d024a94af4323a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdc49bc22c174604e,
                    0x117c78eb0c76e023,
                    0x6ab86dd1e06c7b42,
                    0x025fb549dd9158e9,
                ])),
                Felt::new(BigInteger256([
                    0xc9130da7df194fba,
                    0x048af02866adb856,
                    0x77cadb71cd51eb5a,
                    0x050d1f37448d1f78,
                ])),
                Felt::new(BigInteger256([
                    0xe5203e029a487d6d,
                    0xd91b88236477288a,
                    0x10a1181af5aa6813,
                    0x2ae469791c8b8d6d,
                ])),
                Felt::new(BigInteger256([
                    0xa850280f68306071,
                    0x2861f135b5b36dbc,
                    0xac5eb6d5a020b445,
                    0x247babebc9337b94,
                ])),
                Felt::new(BigInteger256([
                    0x936ef250ed3674a5,
                    0x02563f627b913f09,
                    0x822bd94249741518,
                    0x071a5f10fa4207d0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa133e0484c448d73,
                    0x98c665156a53da3c,
                    0x197e72c44178bd72,
                    0x17f6b7ced749c6e7,
                ])),
                Felt::new(BigInteger256([
                    0x33c861f00f9d15b4,
                    0xc1ba131f368ad2ab,
                    0xe7889267b055402b,
                    0x2bc5d0827b1cf17d,
                ])),
                Felt::new(BigInteger256([
                    0xb6e73300660dc001,
                    0xafe85770385b4f62,
                    0x891eecb70536d410,
                    0x2a237b89d30c7500,
                ])),
                Felt::new(BigInteger256([
                    0x0bf125d953558f5e,
                    0x82049c636be7f065,
                    0xcef6b2111a77724b,
                    0x06c6d9dd10b417e8,
                ])),
                Felt::new(BigInteger256([
                    0x8302c47df754675e,
                    0xec835171c60853b9,
                    0xe7576536dd4c1e34,
                    0x0266164eb5adeca3,
                ])),
                Felt::new(BigInteger256([
                    0x8aed996fa2b8646b,
                    0xdaa6a5afd1dd740d,
                    0xb963c86df5265956,
                    0x26b29cf157059c96,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xd120b3430ccf3fb1,
                0x59c26482de912555,
                0xb2838fd4c85cb26d,
                0x085970977d3ff130,
            ]))],
            [Felt::new(BigInteger256([
                0xd1281ab82f9060e7,
                0xe71a7e30c5a70ccc,
                0xf8bd8b7bb60b97c5,
                0x26e8bb407183ead7,
            ]))],
            [Felt::new(BigInteger256([
                0x54ecbbd7fd72457e,
                0xf48ef898f4499994,
                0x94a5e9b47258ba6c,
                0x108f7c966e9a136a,
            ]))],
            [Felt::new(BigInteger256([
                0xdafc46402b33875e,
                0x00a8ee6184936f58,
                0x36780ae384aac269,
                0x10bcc98289494553,
            ]))],
            [Felt::new(BigInteger256([
                0xe2afe55ddbd40d1d,
                0x46e81912a04f2d64,
                0x7c58d23eacde0b24,
                0x1301c5c35a0e8170,
            ]))],
            [Felt::new(BigInteger256([
                0xf76c85fccd37ed96,
                0xeeb2acd89675bb50,
                0x747549bbd7e049cf,
                0x3006a3690c5a9f0f,
            ]))],
            [Felt::new(BigInteger256([
                0xd241df683c459415,
                0x66de3ed3c5932f90,
                0x1d7b2699e85122c2,
                0x272ffba096a0d5db,
            ]))],
            [Felt::new(BigInteger256([
                0x9ae7ffb3898f1e4b,
                0xdfc44f9d58d143f8,
                0xc933c562dadb2322,
                0x1e21b626d89bb1f3,
            ]))],
            [Felt::new(BigInteger256([
                0x55dde698bb7eee6a,
                0x09347dde20e9e4a0,
                0xe695dfb352b4682d,
                0x1e59267f9eef0b72,
            ]))],
            [Felt::new(BigInteger256([
                0xbba965985efbaaf2,
                0xf8e9fd4174cb2b43,
                0xb9e6b763ff6487c3,
                0x1f93b195dc1e9816,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x600c3c1438bd581f,
                    0x13670a8dcaf4927a,
                    0x285384d7ca143fdc,
                    0x217b188e8db5bae0,
                ])),
                Felt::new(BigInteger256([
                    0x2c7cbdc56fb1a989,
                    0x2ec1df93a6cb2656,
                    0xef845cb9212397ce,
                    0x2f5def0b4c08cca5,
                ])),
                Felt::new(BigInteger256([
                    0x5e760ae52df65cc2,
                    0xc7ce5d3c029580d1,
                    0x67e405aa3370d8d4,
                    0x2f3ad42cab61d51e,
                ])),
                Felt::new(BigInteger256([
                    0x23cd81895dc9493f,
                    0x015622e746a25489,
                    0xfe012473e1ab0166,
                    0x18419938298ec1de,
                ])),
                Felt::new(BigInteger256([
                    0xa91f2763628b2018,
                    0xf086aff30061dc7b,
                    0xb4dedafeaaf846ba,
                    0x21e1f82004a2b824,
                ])),
                Felt::new(BigInteger256([
                    0xce2a96401ebbf749,
                    0x58b7b9b1f6159141,
                    0xf3bbbd74670ae8f4,
                    0x0fb1f6c42ca69629,
                ])),
                Felt::new(BigInteger256([
                    0x2b0f6a8096abcbe4,
                    0xdc2ec633f0447eab,
                    0x9779434870e2326f,
                    0x163088ff96792741,
                ])),
                Felt::new(BigInteger256([
                    0x4ab45a4f81e15565,
                    0x493f3a8e5182205d,
                    0xc5a381489f19a41a,
                    0x2e22d589ece26e6b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa735bc5249e70834,
                    0xc925b0331376897e,
                    0x7279bc01c24f2c57,
                    0x196b4075b2326c75,
                ])),
                Felt::new(BigInteger256([
                    0xa93082e2e4a76ab0,
                    0xb47185b46b1dfa2e,
                    0x3d65f4dcc8a1bc1b,
                    0x1efd9bba8f0f7d43,
                ])),
                Felt::new(BigInteger256([
                    0xf2839f901bfeaef5,
                    0x02060d2b625bea9e,
                    0xd923d808d157a05a,
                    0x19fcae5a5dc87b23,
                ])),
                Felt::new(BigInteger256([
                    0x47918becc8a49cd1,
                    0x7304b36334c6d767,
                    0xd4c3e24f63b7cd14,
                    0x2705924795b8c0d5,
                ])),
                Felt::new(BigInteger256([
                    0xab2a5d9eb472a587,
                    0x77a2c9a4f4d53882,
                    0xd8583bf83d5dfe45,
                    0x2785258372b2a03f,
                ])),
                Felt::new(BigInteger256([
                    0xc346e0dc2045003b,
                    0x5cacb74db97a1a99,
                    0xba6de3cdbd3a03e0,
                    0x24f87d38f50a6ec0,
                ])),
                Felt::new(BigInteger256([
                    0x6c35bf0db3798fde,
                    0xb236818d56b08c17,
                    0xe326229364f93fb3,
                    0x16128fb5605f65ec,
                ])),
                Felt::new(BigInteger256([
                    0x4a5e743f043e4f52,
                    0x855ac17ad0ed5df8,
                    0xfd9d5929a243c3c4,
                    0x0fd3de62ff691bb2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6e06ac782ed1ecfd,
                    0x03a74d1b3649a6ce,
                    0x0dfeea8c9b5c7186,
                    0x035e083ed853e136,
                ])),
                Felt::new(BigInteger256([
                    0xf2ee2240f895c84b,
                    0x0f5997a6e374181e,
                    0x9a68b377bba74642,
                    0x02b3bfd3565ab611,
                ])),
                Felt::new(BigInteger256([
                    0x19e0b862cefe655f,
                    0x7c78f9144115b8f7,
                    0x4279ee851a54f5c5,
                    0x1909b1dd8754e97d,
                ])),
                Felt::new(BigInteger256([
                    0xae8eb548d4b180e8,
                    0xcfa9565e45664498,
                    0x5a3ac146358ca833,
                    0x1990ce75d13a94c9,
                ])),
                Felt::new(BigInteger256([
                    0xec35dccc3bc12719,
                    0x337ed4e83a46d9d6,
                    0x22465c2849293069,
                    0x2c57e5d86c5d9ef4,
                ])),
                Felt::new(BigInteger256([
                    0x6ceee24d7fca2a9c,
                    0x19df8239a7f83a82,
                    0xc5f2774d2da470ec,
                    0x23400a86719c2f48,
                ])),
                Felt::new(BigInteger256([
                    0xfe89823d2115316e,
                    0xca9533e62b813880,
                    0xdfaa01f8738611a6,
                    0x0ab62f6ea9b99423,
                ])),
                Felt::new(BigInteger256([
                    0xc711cebb8c518f4a,
                    0xf65ebf0fac3695b7,
                    0xa58fe28ac16ab7c2,
                    0x146ea4a539aa2c81,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf33ae208dcb828df,
                    0x5102c2b39076dad5,
                    0xbafc9b54d95d0b26,
                    0x07c8a3e031343486,
                ])),
                Felt::new(BigInteger256([
                    0x887f0dc809f3f5bd,
                    0x4c4cfefd9b95b5c3,
                    0x1303a53e2ff7f9b6,
                    0x0a800b1955dece84,
                ])),
                Felt::new(BigInteger256([
                    0xa4b0fd8206b3bc17,
                    0x3bc8d7d5fcfe90f2,
                    0x33a096a6d6b96534,
                    0x23ca0d501b8220ed,
                ])),
                Felt::new(BigInteger256([
                    0x5ae3ada3c85e23ac,
                    0x4f7e07780af22c48,
                    0x00099f0ffdcf04a4,
                    0x1536f4f76b0e7291,
                ])),
                Felt::new(BigInteger256([
                    0x5968a1106e3f583c,
                    0xc501e7452dc61f7f,
                    0xf78a16b485b3e17b,
                    0x016feb4d1c2ccc4d,
                ])),
                Felt::new(BigInteger256([
                    0x9fb5cd38bd7b34e8,
                    0x44017e9f56e6bc99,
                    0x047fa2c5d4c25ee7,
                    0x185950dd184179bd,
                ])),
                Felt::new(BigInteger256([
                    0x25c358d4934a3803,
                    0x6dd36869e8abc7e3,
                    0x26bda39042c379d4,
                    0x052d9a97bfb2463f,
                ])),
                Felt::new(BigInteger256([
                    0x87f9c4e43f6e713e,
                    0x82f4e91c722abf52,
                    0x1ff6973ade98e49f,
                    0x03f7bc407a6f3dc1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa05178a38fea543e,
                    0x8a096215e9e73a2f,
                    0x2c8aebf79c5f680a,
                    0x157f2d22012558d7,
                ])),
                Felt::new(BigInteger256([
                    0xa6ec54954d7c5e0d,
                    0x77d822b099d413c1,
                    0xfb7778e9cb6cbe2d,
                    0x301f66cc23025af8,
                ])),
                Felt::new(BigInteger256([
                    0xa59eb3a5c5956592,
                    0xf768bcc0d0248b01,
                    0x3faaa3308c507f6a,
                    0x2ef8913d95c19c93,
                ])),
                Felt::new(BigInteger256([
                    0xbceb1fc9ff004a4e,
                    0xbfe7925d2d306bd2,
                    0x3e9cb9e03065ba59,
                    0x18db019d8bf0d12a,
                ])),
                Felt::new(BigInteger256([
                    0x00a3c1271fcd3ecc,
                    0x368e92f0fb6b2292,
                    0x69bddf05119e1789,
                    0x2e72b4cba32a6a1d,
                ])),
                Felt::new(BigInteger256([
                    0x01663766c5c22cdd,
                    0xf6bfa8d3834321a1,
                    0x4d084197598abe60,
                    0x013695fcb8faf94c,
                ])),
                Felt::new(BigInteger256([
                    0xe60de0a29ce3d705,
                    0x6dd737628e0ecd43,
                    0x0b9bef5fc09ecc73,
                    0x1e91c38e4ddadb0c,
                ])),
                Felt::new(BigInteger256([
                    0x7cfad5c9db9118c3,
                    0xd9cef5d42fa73589,
                    0x98a32badb64cdc1a,
                    0x0ad4eb08f0bebf42,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf3c0d474e27d67db,
                    0x7220e2021de8080c,
                    0x8045e73a90a032db,
                    0x29f1df390474d44f,
                ])),
                Felt::new(BigInteger256([
                    0x2f3855564be89e91,
                    0x2e183ad7d94ce64d,
                    0xd28c60c685d6f760,
                    0x181538968969253d,
                ])),
                Felt::new(BigInteger256([
                    0xfe70443d4ffcce47,
                    0x7f50bc4335e1bfa4,
                    0xb354121650faa25d,
                    0x0a7769826f5c43e3,
                ])),
                Felt::new(BigInteger256([
                    0x37d6d1b969f9b461,
                    0x16f7604594b1076a,
                    0xcd161b05595b90e9,
                    0x1ac006fa78a4363c,
                ])),
                Felt::new(BigInteger256([
                    0xcc3f0cd8dc15bb68,
                    0xc9c8035c85a8e083,
                    0x287c56f523546abf,
                    0x1aa5e2fc51e8ebcf,
                ])),
                Felt::new(BigInteger256([
                    0x80a9c3c5fc2e6d19,
                    0x35f2d24057d7ee27,
                    0x941b11befdd5b568,
                    0x266f0cf5b4d89ab3,
                ])),
                Felt::new(BigInteger256([
                    0x6d29a5d44bcfe823,
                    0xc6ca86bcd9ee5362,
                    0xdd39ace42499d7c1,
                    0x0dade98346885ecb,
                ])),
                Felt::new(BigInteger256([
                    0x6804ba24a5480308,
                    0xfb483f23deb980cb,
                    0xfa652748873383d6,
                    0x06a097ba4025659e,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x96d21be6ef004278,
                    0x77507de523854bfd,
                    0x4656adfe45bef3c3,
                    0x211b0bb7729fda61,
                ])),
                Felt::new(BigInteger256([
                    0x25f1b60150d8aba2,
                    0x8d0701eb01365cca,
                    0xe960669c33f6c027,
                    0x16ce1ffc6f27a0aa,
                ])),
                Felt::new(BigInteger256([
                    0x0ae040aa10baa3da,
                    0x6b43da3f2039f50d,
                    0xeb265a535bd1c70c,
                    0x0588d41cb3e088e6,
                ])),
                Felt::new(BigInteger256([
                    0x3e1e7d408fd4d505,
                    0x01968ad65e6fc4e1,
                    0x978190256afeff75,
                    0x01c44161b8bd53d3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2f61803470cb3c80,
                    0xf8d81d9f6ebd5a5c,
                    0x41d3e026b310d349,
                    0x09babdc52fb878bd,
                ])),
                Felt::new(BigInteger256([
                    0x1baa1406acf3c695,
                    0xd66150c3c792eb81,
                    0xf5ff6aed6142a0fa,
                    0x16953968e638fb82,
                ])),
                Felt::new(BigInteger256([
                    0x0bdbd84b6666f954,
                    0x34d9136d54e6b41f,
                    0x7076371d22beef69,
                    0x2d6979122d14737f,
                ])),
                Felt::new(BigInteger256([
                    0x21da1e608cb3372c,
                    0xf6c97758d1a36955,
                    0xa2871fbc7dbea13e,
                    0x2702a74e0599334f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcedef4ade9f076c4,
                    0x359826a23bad9235,
                    0xe8264227bedae270,
                    0x1ad73be6fdbf1ddd,
                ])),
                Felt::new(BigInteger256([
                    0x500c1b0e2dca0f54,
                    0xd4ab34dffd9bb8ba,
                    0xaf083aecdae7e090,
                    0x1e26547f69b88809,
                ])),
                Felt::new(BigInteger256([
                    0xcce423e684fc5489,
                    0x99c46acbd2b2865a,
                    0x0935a40658f42b82,
                    0x0e5c9327267aad77,
                ])),
                Felt::new(BigInteger256([
                    0x913cbeaca9de73eb,
                    0x19e30669928ba4da,
                    0x32644ab68371f740,
                    0x1047976909c84c23,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe331658b6365e012,
                    0x7f16c25aadcf943a,
                    0xa91319ab3a541880,
                    0x150efed59515117e,
                ])),
                Felt::new(BigInteger256([
                    0xf3c05c81e60a6232,
                    0xd913ff3d35bad7ee,
                    0x77f171e1cc5ee2e8,
                    0x10b053cfffaddb1f,
                ])),
                Felt::new(BigInteger256([
                    0x21f6af8e52278532,
                    0x33a77dd40ec76295,
                    0xb9129d1be9560fde,
                    0x1b492f427d9aab35,
                ])),
                Felt::new(BigInteger256([
                    0x1f15509107886a17,
                    0x0f41a481b8c2f914,
                    0x221171052f59b974,
                    0x0789256c9215c6bc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8e38d9cfefebb1ef,
                    0x99318a9797dee90c,
                    0xc91a524f5123ba00,
                    0x1d62d186c4ab548d,
                ])),
                Felt::new(BigInteger256([
                    0x6be2fc3296c9e30f,
                    0x95ffd60c057c6db3,
                    0xb9cc3d34a8d13a6c,
                    0x052307f748303318,
                ])),
                Felt::new(BigInteger256([
                    0x9d8c3fd36c52d2fc,
                    0x836314fe106af005,
                    0xef65da5c4a3d68ca,
                    0x0ed4972eee80e5e4,
                ])),
                Felt::new(BigInteger256([
                    0xc469276bc90cb661,
                    0xb0b3ae41916b5062,
                    0x4a4c90bb23be55a8,
                    0x19356149effd9bd2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcc3b9a067d8674dc,
                    0x4dfe9cb1baae7e40,
                    0x5c31ff1c72e6d354,
                    0x0e30d70dbf0f03c0,
                ])),
                Felt::new(BigInteger256([
                    0x69b83374df6378b7,
                    0xda2d07663dfe622b,
                    0x818e94744878cc2a,
                    0x0a1177ab98fa665d,
                ])),
                Felt::new(BigInteger256([
                    0x24a0f1eb277e85cd,
                    0xaaf2657c1178ba9c,
                    0xb16c36d4a518935c,
                    0x2cf161818c31dc74,
                ])),
                Felt::new(BigInteger256([
                    0x83c88e64a4870f1e,
                    0xffec04ec8f0b94c3,
                    0x59cdaa3658147ee7,
                    0x28a2d24e2e77aa86,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x814940c60f23be80,
                    0xab78a5added5ab05,
                    0x312829cec0edba39,
                    0x2bd2b40db4d826af,
                ])),
                Felt::new(BigInteger256([
                    0xf3ecbfd83f254252,
                    0x8d599f4cd2ebe5ee,
                    0xebe3dcb4c4527b02,
                    0x07c894946d75984b,
                ])),
                Felt::new(BigInteger256([
                    0x49e9fbdcf53526bc,
                    0xaae3d25fa5e70d2e,
                    0x10f8d026a6a28cb3,
                    0x14932634a8360636,
                ])),
                Felt::new(BigInteger256([
                    0xebb330c802e1c9b4,
                    0x7714bc9ce8182d96,
                    0xea7870aacfb597d3,
                    0x122e992d2a5ca7fa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa56679c496b8104e,
                    0x0e82085b72181c90,
                    0xd2e863a3b11bca26,
                    0x112444bceddf91bc,
                ])),
                Felt::new(BigInteger256([
                    0x2ffe3764fb35111d,
                    0x19d1171e47595933,
                    0xd639941bfa67b065,
                    0x1b806216641c74f0,
                ])),
                Felt::new(BigInteger256([
                    0x554973b410f90dce,
                    0x16937a6ea9bea479,
                    0x9dfbbafe606df552,
                    0x0bf7ebe5ca14abc1,
                ])),
                Felt::new(BigInteger256([
                    0xcabfd92253744f04,
                    0x1fcb9956a6058659,
                    0x55e4a2bdd7298ac3,
                    0x2a57bdcb15a0cdfa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x739e8fccf4c22c46,
                    0x6bf64196304824b2,
                    0xd1ef1c5eae2f6fe8,
                    0x198c4ea0de9351a7,
                ])),
                Felt::new(BigInteger256([
                    0xb48a404e8f7067fc,
                    0x1868d2557b9e9848,
                    0x7f9bfece348d3bd5,
                    0x200880356cf986e4,
                ])),
                Felt::new(BigInteger256([
                    0x8a6647ee6cca2787,
                    0xfa565c124c042057,
                    0xfb34d767d5e4ed3e,
                    0x0e4daaa187d080bb,
                ])),
                Felt::new(BigInteger256([
                    0xf12d24445ab80654,
                    0x8452dc89282074b2,
                    0xa511f6c6b4ced7be,
                    0x290e4b367859626c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9a9d246b143190ec,
                    0x4cd69cb7575d6c2a,
                    0xe722c55358620e24,
                    0x0279a29c576eb7c0,
                ])),
                Felt::new(BigInteger256([
                    0x5f74f5f68f5c771e,
                    0xb77262dca7567749,
                    0xfc78421d8839981f,
                    0x1939e893efc222a3,
                ])),
                Felt::new(BigInteger256([
                    0x16a4b3fa862ed7e3,
                    0xe7cc0ae735cccb1a,
                    0x3456339458401152,
                    0x1cbf260e66af0b2b,
                ])),
                Felt::new(BigInteger256([
                    0x12ad9f6f655ab3aa,
                    0x5cbfcbbb8d830d2f,
                    0xb4559d67ac628c5b,
                    0x264ebb0221e53151,
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
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x22564a8f8f3cd05c,
                    0xcd8ee614fd63b5dd,
                    0xb3ff0abacf5a7163,
                    0x0419ff77206f2ff4,
                ])),
                Felt::new(BigInteger256([
                    0x408bbbbe90ef5913,
                    0xa8250dbb15372781,
                    0x69e0c44d343d364d,
                    0x07a9898a71e309c0,
                ])),
                Felt::new(BigInteger256([
                    0xe0b5488eb2075dd2,
                    0x31dae88a816b1309,
                    0xad5a9e5ca47ecabb,
                    0x0d7b6b39dcc47f08,
                ])),
                Felt::new(BigInteger256([
                    0x1d8c9dc0971827dd,
                    0x45f9b219f69a1807,
                    0xee0473df8e4d0578,
                    0x2dc9f73a098982dc,
                ])),
                Felt::new(BigInteger256([
                    0x65ed2d9201530504,
                    0xbf7ddb9aea756270,
                    0xede06afb5fd858ac,
                    0x00a31edb463cb016,
                ])),
                Felt::new(BigInteger256([
                    0x524482a3b245db1c,
                    0xfb1d40f203ff2063,
                    0x4966197c405fdc48,
                    0x27c69613a46d9aa9,
                ])),
                Felt::new(BigInteger256([
                    0xcd34150ee31138ca,
                    0x419bfb785273905e,
                    0x41f9ff1fb8df706c,
                    0x04137704624e1f4b,
                ])),
                Felt::new(BigInteger256([
                    0x25c3db895228764c,
                    0x6beae38989b27bdb,
                    0x71d72029fc4b9010,
                    0x157ca86a1832887e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x994e9ff71d27da64,
                    0xe59b8ff84eed0c8b,
                    0x35e316c6a2c11b3e,
                    0x2b4c6dffff3d7141,
                ])),
                Felt::new(BigInteger256([
                    0x746e21027bc93b9b,
                    0x91c31e2c0a0ac415,
                    0xdaa50e4c8b00081d,
                    0x09c1017ab3ae3e40,
                ])),
                Felt::new(BigInteger256([
                    0x5b7aca138e3b9d98,
                    0xa9d0b8f82fdbdce1,
                    0xbdc6e5b8653f8aa1,
                    0x243447e0b615a581,
                ])),
                Felt::new(BigInteger256([
                    0x7e0a5888bd2a6442,
                    0x4abd795ac4b069bf,
                    0xb0c54f011f873330,
                    0x1892e172a91b6ec8,
                ])),
                Felt::new(BigInteger256([
                    0x7c81e77070b61f14,
                    0x1ea304de39cf40f7,
                    0x4ce0dd23645f2037,
                    0x2898e5c18d994da7,
                ])),
                Felt::new(BigInteger256([
                    0x31fb724e0d6d5273,
                    0xaed1983404c1c786,
                    0xf727cafa94a175ec,
                    0x118acfc569122fba,
                ])),
                Felt::new(BigInteger256([
                    0x630283bee30c71b3,
                    0x22a2b914ae2f2dc2,
                    0x96e2befcf5067e9e,
                    0x203af45f5bfa808b,
                ])),
                Felt::new(BigInteger256([
                    0x88f904e15a9ce24b,
                    0x55069a2c2ae89d8f,
                    0xe8faad50636217b6,
                    0x045613d87ddd0f2d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd633531a4c433052,
                    0x1cb5974c08efd49c,
                    0xcf1f2f67f478e9f1,
                    0x170dc88bef3afa43,
                ])),
                Felt::new(BigInteger256([
                    0xe89e21b000129dff,
                    0x970143735c912a0a,
                    0x997afe491b24b8cd,
                    0x055ee429e1ac9943,
                ])),
                Felt::new(BigInteger256([
                    0x293e80f50eff9369,
                    0x493c45cec2da8a1c,
                    0xbed993e7750e001e,
                    0x0dc40dd0c63d93ab,
                ])),
                Felt::new(BigInteger256([
                    0x47447ecd6c327cb8,
                    0xbcc5e20f53fe25f2,
                    0x9087d8a0d5fc6096,
                    0x1c649910893f04b1,
                ])),
                Felt::new(BigInteger256([
                    0x2bdd0f314fd09b84,
                    0x96d2d45c0b2db537,
                    0xd67367c08390860c,
                    0x20cf3c3d94e011ba,
                ])),
                Felt::new(BigInteger256([
                    0x7a7804919b419c70,
                    0xef89b053814dd44b,
                    0x3ca9d26c995daa25,
                    0x12693a38586e4951,
                ])),
                Felt::new(BigInteger256([
                    0xcc30e5cedb43081a,
                    0x172688afcbcba1cb,
                    0x52888be4ae978054,
                    0x228f2d45e9483579,
                ])),
                Felt::new(BigInteger256([
                    0xc1f4510393c10f3f,
                    0xd8506064666ad4d5,
                    0x1aee42518b14d6c6,
                    0x29847946dc50bcab,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb69a9e731336507f,
                    0x30d16d065fb4b1d9,
                    0xae8ac225d8301fb5,
                    0x2a0a4cd97dcb8552,
                ])),
                Felt::new(BigInteger256([
                    0x17c8217fc7ed3781,
                    0x67f81964ede7781a,
                    0xea38eeccaeadd018,
                    0x2af9a6dad2ff91f3,
                ])),
                Felt::new(BigInteger256([
                    0xb554635972ba31d6,
                    0x900420f941965927,
                    0x8d6ca2eb266be48c,
                    0x07815e8fa633e362,
                ])),
                Felt::new(BigInteger256([
                    0x810ca19b5a3debeb,
                    0x0d49fe5fea71d257,
                    0xb4e1e868adbc0a6e,
                    0x2f5445a177d4ae1a,
                ])),
                Felt::new(BigInteger256([
                    0xfd7ea47f20c585bb,
                    0x88919b0fdbe6184e,
                    0xd5a6a3c1c2e8dce7,
                    0x2732bf5796dd6085,
                ])),
                Felt::new(BigInteger256([
                    0x2dedfcbe7699fc3c,
                    0x1967b91aaf784541,
                    0x360a716e5b0fd8cc,
                    0x22ddaf96ce61c027,
                ])),
                Felt::new(BigInteger256([
                    0x6d8c865ed1d8190c,
                    0x4251b94ca889f82e,
                    0x38da063a8687b7f8,
                    0x185d72e4211430dd,
                ])),
                Felt::new(BigInteger256([
                    0xa62287ca8f420f6e,
                    0x9f59f593eb21a00f,
                    0xe2989960587cf601,
                    0x113853e6d8694ea4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x98ba29f44ec24ef9,
                    0x37a5d2adcf6a2e15,
                    0xd1fb76db4c9a8b27,
                    0x2f4601d939716169,
                ])),
                Felt::new(BigInteger256([
                    0x339007d18f46a81d,
                    0x13ab7f6845022288,
                    0x9ae28c53f72b47bd,
                    0x253ae1fe81acef41,
                ])),
                Felt::new(BigInteger256([
                    0x7d37bde357ec00a4,
                    0x87b7c0592b200b51,
                    0xcca86c009b62dec5,
                    0x06e29d4d679a95b5,
                ])),
                Felt::new(BigInteger256([
                    0x7a387ed0790c2222,
                    0x9683a9726a8efae6,
                    0x7dca53a7820f34ab,
                    0x2c1130d4d01b9889,
                ])),
                Felt::new(BigInteger256([
                    0xc735f5e3b4c4a5c2,
                    0xae5704457125a8d9,
                    0x1f911a85a43f5edb,
                    0x030e4c38adcc39a0,
                ])),
                Felt::new(BigInteger256([
                    0xabe05201be8caf8a,
                    0xa678d0d3b41ad431,
                    0x8681c7f31f00f1e3,
                    0x23b902b6e5cb9b0f,
                ])),
                Felt::new(BigInteger256([
                    0x0a60cb480103a11b,
                    0x560db30b9452d89a,
                    0x85306bb0b6a0e75b,
                    0x0c8f1d440c220c82,
                ])),
                Felt::new(BigInteger256([
                    0x5da20258962ddfbe,
                    0x7f61c35489bb6a44,
                    0x8df650aa47a5f31c,
                    0x2441b265f0135551,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0c8d0103e7ac14ba,
                    0x5fe61624e7ce7c65,
                    0x8774417f78577376,
                    0x27e4c39149ca2b73,
                ])),
                Felt::new(BigInteger256([
                    0x5faefaa5fa545ca6,
                    0xdc7eb1f8f99c2ad7,
                    0x27be9849ca1d0f82,
                    0x00291c80c0706759,
                ])),
                Felt::new(BigInteger256([
                    0x325a1abc8248c2e4,
                    0xbb7d3a1d21f0b388,
                    0xb4a1e24d3b1839cc,
                    0x090ca8095f3bddb4,
                ])),
                Felt::new(BigInteger256([
                    0x03bdf01f1b1a1add,
                    0x6f9d1f48582a1a6e,
                    0xb4e2b31110d2435f,
                    0x1ac8fc48b5b1e9c1,
                ])),
                Felt::new(BigInteger256([
                    0xd8f8f8bc70535b7b,
                    0xedc9069b932c090a,
                    0x62f3e20af7370f97,
                    0x07f78d3f1e8dee26,
                ])),
                Felt::new(BigInteger256([
                    0x61915ff29af0e8c2,
                    0xe9341208b2c6a6b5,
                    0xa0b4d38030b6ffd4,
                    0x167f2698b12a4036,
                ])),
                Felt::new(BigInteger256([
                    0xf6983db48ec92640,
                    0x1b957f90fd98bc05,
                    0xd30efb1c9c268306,
                    0x0bbb4ab4e2a8f77c,
                ])),
                Felt::new(BigInteger256([
                    0x38be52db0bd678d8,
                    0xbf73ea4736045f99,
                    0x3f2671b62207143e,
                    0x121d0ea86827ab07,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xc9a203bc07eb69b2,
                0xd9b07a543af39828,
                0xfa0eb95cbf05220e,
                0x0ed1f2bf6d33b79c,
            ]))],
            [Felt::new(BigInteger256([
                0x008072b95fdf3907,
                0xcbd924068bf6ce37,
                0xda301680b1ce5431,
                0x13f37aa8863bdabb,
            ]))],
            [Felt::new(BigInteger256([
                0x40eb66386e185145,
                0x266962263615ab98,
                0x1a78261af4a78d66,
                0x273d6c83b688ff58,
            ]))],
            [Felt::new(BigInteger256([
                0xdbdd3615caa33446,
                0x0392795c42a2fd45,
                0x43d853f79de16c5e,
                0x182d58e1c341be66,
            ]))],
            [Felt::new(BigInteger256([
                0xd230e1ecd1223511,
                0x532dad6ddc96ff78,
                0xb65ce392610d2289,
                0x00248f41203b5b81,
            ]))],
            [Felt::new(BigInteger256([
                0x6b30b4a522afab17,
                0xf79b195b489fd1c9,
                0xe6090a56ed97b8f8,
                0x0683d5935acc5686,
            ]))],
            [Felt::new(BigInteger256([
                0xbc20c8f028c79325,
                0x311ec17f470b2518,
                0x936a78216f59323b,
                0x1409b6184313fc46,
            ]))],
            [Felt::new(BigInteger256([
                0xbeb04d72c13960c8,
                0xb225a718eed62999,
                0x53d128d47476c4db,
                0x02e06f99c025891d,
            ]))],
            [Felt::new(BigInteger256([
                0xa47344681c6d2558,
                0x2f6e2a2158f5baf8,
                0xf5115841f79165bd,
                0x0ff40c3526081092,
            ]))],
            [Felt::new(BigInteger256([
                0x13fce8ffaa9d0436,
                0x23a8b621df0e4bcd,
                0x8257a88f8c93c992,
                0x00a47c17b4359e11,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
