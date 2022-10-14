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
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 48];
        for chunk in bytes.chunks(47) {
            if num_hashed + i < num_elements - 1 {
                buf[..47].copy_from_slice(chunk);
            } else {
                // If we are dealing with the last chunk, it may be smaller than 47 bytes long, so
                // we need to handle it slightly differently. We also append a byte set to 1 to the
                // end of the string if needed. This pads the string in such a way that adding
                // trailing zeros results in a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 48];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 47 {
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

    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger384([
                0x90820ef352245e6b,
                0xc58e7fddfcf17fa8,
                0x2f25ba323d322b8b,
                0x25e50596504158f7,
                0x4543634b9f7119bd,
                0x01958b92ed1e48cf,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x55cf094222036f3e,
                    0xf2bcc44cd94d2ae8,
                    0x26332c81cd148c10,
                    0xdca4a85652aef69b,
                    0xde583ada6814c338,
                    0x009f1396e6ad8402,
                ])),
                Felt::new(BigInteger384([
                    0xa8db75f445ff45a6,
                    0x2c924ba732ec4c11,
                    0x9230aab0bc638831,
                    0x7cc94f6fff226d56,
                    0x3b1162bae6d43c74,
                    0x00e4dc0af8e05278,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf4ae0b7d1d5b6c05,
                    0x40deb1c98872bd1b,
                    0x914b04303092a42d,
                    0xe2c5b4748ba60173,
                    0xab6e31bcdccd837a,
                    0x00c30b31fccf4439,
                ])),
                Felt::new(BigInteger384([
                    0xcea9c0c81dd986b0,
                    0xcde60d7025dc983b,
                    0xe6f0ec003f704769,
                    0x5e4c2dd0c1b275e4,
                    0xf05e8085f3b98abe,
                    0x00f3d8be3c575bac,
                ])),
                Felt::new(BigInteger384([
                    0xb4033c58e9d2db13,
                    0xe6186837b9488e06,
                    0x61c4212e55c5f315,
                    0x0387723a89db6666,
                    0x3cc5c175f62ae231,
                    0x0113a6210bf06214,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x608e1f9499c4c155,
                    0x27859cc191930ad4,
                    0x2dfaaebc03ebae70,
                    0x2367c55763c1d96f,
                    0xa259105421874610,
                    0x0098837dec7ee4a7,
                ])),
                Felt::new(BigInteger384([
                    0xfa8e082d1a945354,
                    0x9a29ce27d455116e,
                    0x1b84f0099fe2b716,
                    0x145be375d1da0e1e,
                    0x00b66bd13b890845,
                    0x00d3c5db3a5fd1bf,
                ])),
                Felt::new(BigInteger384([
                    0x10c1b04d7e554968,
                    0x55c553841d93b589,
                    0x770b16104fb55af2,
                    0xb35c6477a34fe3ec,
                    0x134c7f16e4c9a7f4,
                    0x013a6fc5ae2421b9,
                ])),
                Felt::new(BigInteger384([
                    0x4e4cf2a41a1f55b9,
                    0xe1b59692ca1b93d9,
                    0x0ceb54655257b010,
                    0xe72721b984b30b8a,
                    0x032ffc30bb79598e,
                    0x00ed81ea0a7e7171,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd4d0faf9a6e57555,
                    0x44ad1ed06b15c906,
                    0xa37b54b6088dffda,
                    0x7ae07a8a2c0df85b,
                    0x2a6ed5a299b60dce,
                    0x01710b29527aae31,
                ])),
                Felt::new(BigInteger384([
                    0x64c9e52800280e36,
                    0x3561c9986d2afdcc,
                    0xe0851cfad25175d3,
                    0x5a33d5260a9aa7f5,
                    0xa30cf246d826c9c6,
                    0x019b5ba5c03a9e00,
                ])),
                Felt::new(BigInteger384([
                    0xa7cffdc9c0e9660d,
                    0x9c663ded7bce4f5c,
                    0x73ae5671395743fe,
                    0x522207d32e1787dd,
                    0x0cb6a4b31f1d6929,
                    0x0018f19f25465112,
                ])),
                Felt::new(BigInteger384([
                    0xdcfab6e3a2faaa19,
                    0x2d17cb0bf248a387,
                    0x8666bc49aa4928ee,
                    0x8acf9058a3017378,
                    0xe7366c30687bbdde,
                    0x0012533e75fb69e9,
                ])),
                Felt::new(BigInteger384([
                    0xdf9f246aa7feb790,
                    0x25511c19df8e366f,
                    0x8c3b0663234fef07,
                    0xd436933a4b9f1c46,
                    0x308aa314b360f332,
                    0x007097abb00e7601,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xdaa83ddac2d4b5db,
                    0xc246b25603da39ac,
                    0xcf4c48edea0e6712,
                    0xabb1785f27fdceea,
                    0x22be925340e90f82,
                    0x002d5a4078252422,
                ])),
                Felt::new(BigInteger384([
                    0x74e1a01e47dfafee,
                    0x5b045f7a65e76b5e,
                    0xef6a0ae675864f51,
                    0x866cf3f564c59ce4,
                    0x98916bdcc48934d6,
                    0x010db92cce243c91,
                ])),
                Felt::new(BigInteger384([
                    0x6324a5126b5be80e,
                    0x56c91d44a98d9c6c,
                    0x325d5200af20134e,
                    0x7c0d4ccdd5df4d89,
                    0x3839150a3d0ce875,
                    0x00956f33b16f215e,
                ])),
                Felt::new(BigInteger384([
                    0x4486a96f82d850d1,
                    0x19d18a38a30c4682,
                    0xcb9d241965fbb2d0,
                    0xd7b644f43d31dd9e,
                    0x3b010379528f63cd,
                    0x00ed4ee2fb6f679d,
                ])),
                Felt::new(BigInteger384([
                    0x187b5db7453e2c01,
                    0xa95147327408b0db,
                    0x91f0cb4bb4f742f5,
                    0x71eb7a88e907ce21,
                    0x22cc7e95c5d26f15,
                    0x0092315481211e14,
                ])),
                Felt::new(BigInteger384([
                    0xfe18855a6af05d35,
                    0x392659332e6415ad,
                    0x9cc93376d1a62dd4,
                    0xc357872d01e60edb,
                    0x1cac3023305adc2e,
                    0x00cf73f07f46ac82,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x28c7cec2422cb4cf,
                0xaa72d241f922abb4,
                0x30f48e1c51b93f88,
                0x504b1e2af978ac2f,
                0x9cec476c49b9d6bc,
                0x00e9f4fe3363c8a3,
            ]))],
            [Felt::new(BigInteger384([
                0x3fa7492c2b251ff9,
                0x528c9ca9a0fbb7dd,
                0x7a202cf556148623,
                0xa3c060de6297d717,
                0xa6f6564cd920888d,
                0x000f9e4965ac22d8,
            ]))],
            [Felt::new(BigInteger384([
                0xf7b57d0bccd2c3ac,
                0x0e338c8844219909,
                0xaacdd1795f8f9476,
                0x939178137b2e3bed,
                0x72180c3c883b0758,
                0x00979051afaff4c8,
            ]))],
            [Felt::new(BigInteger384([
                0xebe2e811c9ca10ec,
                0xce96949817f17377,
                0x59121e8cd6cd94cb,
                0xe56772f33e7649b2,
                0xeb3ccd4be7e65c8a,
                0x007288fdcdc966bf,
            ]))],
            [Felt::new(BigInteger384([
                0xc5009ae05eaaaa98,
                0x14288b69605f2181,
                0x57dfd3c73f819155,
                0x5142ad4406412ff1,
                0xe1299e069be337ad,
                0x018dbe40e901e756,
            ]))],
            [Felt::new(BigInteger384([
                0x6a27cd9bc5792d3d,
                0x191839a78d96c71f,
                0xca48981945859931,
                0x892074fcdedee75e,
                0xf29813e310df32e9,
                0x014797cf8e26e4f6,
            ]))],
            [Felt::new(BigInteger384([
                0xae5ddadc2947b0b5,
                0x2a59bcbb09f66ed6,
                0x1e6534f60fdece64,
                0xae4c6fb72c736c9f,
                0xf622c476eca83be4,
                0x0023574ce702a958,
            ]))],
            [Felt::new(BigInteger384([
                0x14ad93ed27f1c6dc,
                0xc6771bce89c26655,
                0x987528c8c811424a,
                0xb233e1bc3dc1bd06,
                0xa31b48babbf9a498,
                0x00c69a3f5129e1eb,
            ]))],
            [Felt::new(BigInteger384([
                0x750b8199f1b8faf5,
                0xf0368adab38cd3f1,
                0xcf5b3dd5449180be,
                0x8d05dcdad41a6297,
                0xc911f03b7a63e7d9,
                0x0129559535fa03c4,
            ]))],
            [Felt::new(BigInteger384([
                0x8e1252eaf8140e23,
                0x7fea7f0c48f94d79,
                0x3d7d8e779bef6b5e,
                0x7af6310ac8bb7161,
                0xa97cc813b1432fd4,
                0x014bd77802c406a3,
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
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger384([
                    0xd894b7aa93c0ad82,
                    0xdc409110188b0b33,
                    0xba59547d18a3a296,
                    0x5dc1aa98daf96a8c,
                    0x98a2f830f2184516,
                    0x0184d0ed3e5b7e50,
                ])),
                Felt::new(BigInteger384([
                    0x165af7d5d64df2cc,
                    0x88c7b2944bcd051d,
                    0x1f8b04c8649f1473,
                    0xa4b9cb3fc1103ca5,
                    0x94db72d5dcf0a8d9,
                    0x007cd832834cfb4a,
                ])),
                Felt::new(BigInteger384([
                    0x6b28bb49ad3dc7d6,
                    0x13f16e81f5e44495,
                    0x1b5a09c0e8aa00e5,
                    0xe423204d5887dc64,
                    0x5e67e077b29d47e2,
                    0x015761ad5ba7ee2d,
                ])),
                Felt::new(BigInteger384([
                    0x58dc88ee422369bc,
                    0xa28357e5f83740f5,
                    0x443e76c4c001f30d,
                    0x338741a82ccc1e4b,
                    0x3eaaab6546fc225e,
                    0x009be29469758fcb,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x42a8215f2c0b3676,
                    0x681a8ba521310cc2,
                    0xe55dd770d61c5839,
                    0xb2cea4632e1c1fe8,
                    0xa048154a4b2b7556,
                    0x00aac8177f591285,
                ])),
                Felt::new(BigInteger384([
                    0xd5f05617b831638e,
                    0x9872e8c7039178c3,
                    0xc2d07773bc67982e,
                    0x583cdcaf7ecf3ca7,
                    0x72064ee642bc9e3b,
                    0x011c009a23a22b5f,
                ])),
                Felt::new(BigInteger384([
                    0x0caaddd4afc99a51,
                    0x0927e335c689f2f0,
                    0x9aacf7e3bcfa7f3c,
                    0x8205015a72b878ce,
                    0xcd253e9937d1ce29,
                    0x0055db2caa5503c0,
                ])),
                Felt::new(BigInteger384([
                    0xd3a559ed13123af3,
                    0xa4ec305c63a31068,
                    0xb8d52fa11b1b80cb,
                    0x5259318e6d06063c,
                    0x590115b5260736ee,
                    0x0095ed1f1b72d605,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x4dc79c09ab3e9161,
                    0x5e1c0a71df06a826,
                    0x7fa552b5366387a9,
                    0xbb1381032d5e8022,
                    0xca9607a3e0fb3594,
                    0x01699e65b80e725f,
                ])),
                Felt::new(BigInteger384([
                    0xee4bb37d914c4b97,
                    0x47ab9ddaa5fb63ed,
                    0xbae20f8cc7ef297a,
                    0x7712ebea6132ee9f,
                    0x17989c67ed97a206,
                    0x01a02fd4b20e0890,
                ])),
                Felt::new(BigInteger384([
                    0xc2753df886e1ecf6,
                    0xdd5f79fcdc65f026,
                    0xed4cd098bbc83245,
                    0x24b61576015b1b25,
                    0xde8f80f27a872d63,
                    0x00b01d2301c97241,
                ])),
                Felt::new(BigInteger384([
                    0xb0af6b215cfe3f72,
                    0xc55d2cd4a89bfc18,
                    0x1b56124d015ba722,
                    0x14a6bf337c53dd88,
                    0x0f56cdc774a059b7,
                    0x01602225a03b4049,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5390e49b13a63b09,
                    0xd5acb5e1fd78aa9a,
                    0x421f3fc7fdc13351,
                    0xa80d831b8a2d079c,
                    0xa9bc862ee42039a2,
                    0x00e90937875985bc,
                ])),
                Felt::new(BigInteger384([
                    0x7802f1a417e4dba8,
                    0x45ff9da3983801e1,
                    0x1c01dd8b91a50dc9,
                    0x9facdac4625f98fa,
                    0x87c092a004af8b9c,
                    0x01360e5c193d4af1,
                ])),
                Felt::new(BigInteger384([
                    0x546250ecc3886083,
                    0xe3e7dd7db3417524,
                    0x81c5e271f4aa0811,
                    0xe095d3bb7debc3a6,
                    0x62fc6cc1f0c59def,
                    0x006fc834bfde904c,
                ])),
                Felt::new(BigInteger384([
                    0x478a9cfda227a1d0,
                    0x4cea19c211484a7c,
                    0xf3b0b0fa263bf88d,
                    0xe07186a6e28c8cdf,
                    0x0227711c2d367407,
                    0x00860b0c14ba5f14,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xbc0db415c6e3ee86,
                    0x36a2365c6049a591,
                    0xa148d63141a6f0af,
                    0x56c0b59964d818a6,
                    0x8fc86d43e113af70,
                    0x01ac3c0790b8d030,
                ])),
                Felt::new(BigInteger384([
                    0xa6d370786073692f,
                    0xa357c4fca36b7c0d,
                    0x6d39e46646033bde,
                    0x0ac3aed082c2e476,
                    0xd1ae9ee711744c40,
                    0x00ec3ed02f9f48a2,
                ])),
                Felt::new(BigInteger384([
                    0xd6daf36452ebf255,
                    0xfee97538430f219d,
                    0x6f5a31409d304fab,
                    0xb762dee95f59b5c3,
                    0xd7b074f823513527,
                    0x008ca7ce5f6773a9,
                ])),
                Felt::new(BigInteger384([
                    0xedcffa249405fc07,
                    0x425d6ef76bba392f,
                    0xee32a376b8aa92f4,
                    0x0410d0e89343d542,
                    0x23a2e20e5dea9c6a,
                    0x00ea6325aebbb6bd,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xf822a8c4837fb065,
                    0xe392b541273fa7db,
                    0x287ef8fa1b08c4a6,
                    0xa8e186ed8ce70052,
                    0x5e5a6cb7d8ebcb0b,
                    0x013a11016d9896eb,
                ])),
                Felt::new(BigInteger384([
                    0x2ec3cc88b1edb6df,
                    0x267f4b6daa8c6984,
                    0x410cd8a14f125dff,
                    0x8182dc9e7033eaed,
                    0x15e035000aa80937,
                    0x012e8af022c28088,
                ])),
                Felt::new(BigInteger384([
                    0xf508078cc314353a,
                    0x3c9f225db5e41295,
                    0x73708e75f75f43d4,
                    0xdf600eab0ddf296a,
                    0xf9f3cb8b11b90f01,
                    0x01111466c2e7304e,
                ])),
                Felt::new(BigInteger384([
                    0x7485bc28f97e51a2,
                    0x3d8a226add7f68cd,
                    0xbe9925e3e65f93d1,
                    0xba8e4894141358e2,
                    0xf8b5e9289c66e040,
                    0x0186bc10c9f77c3e,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xfe7bd1aa972a1b21,
                    0x09473a26c8abb627,
                    0x435b6977a45cbbf1,
                    0x4a7ecddaacb23dd7,
                    0x472331adaa8b1856,
                    0x00f156a9887b8cd0,
                ])),
                Felt::new(BigInteger384([
                    0xc45f300266ccee79,
                    0x512b5aef0c56bd58,
                    0x016877a589c9b280,
                    0x8b40bdb6b338a69b,
                    0x7caa3b8aa73f4de5,
                    0x0051caf9d18fdc5c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x1ae36ddb4f4cf4f9,
                    0x9566ed96ac16d171,
                    0xc89a9361f4c3ba34,
                    0xbf6fcb510e48952f,
                    0xc9e6d0db65ce5a7d,
                    0x01814009cdb7d7df,
                ])),
                Felt::new(BigInteger384([
                    0xdd57e138e3e2b0ec,
                    0x5c8a409cd8472346,
                    0xa258013879be44a5,
                    0xc4b27540267d17dc,
                    0x36a8793701099b97,
                    0x0163eaa770140f58,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x03365befaf277f39,
                    0x96967a4750de5e87,
                    0x382d440e65b1c3a7,
                    0x14101536180479f8,
                    0x1ac1d58777816bb7,
                    0x01457ce904e1c1db,
                ])),
                Felt::new(BigInteger384([
                    0xbefe9e562d7d7466,
                    0x8f30120bb36e34ea,
                    0x78ef0de2acb03c7c,
                    0x6c2fdfd4f3d9bc6e,
                    0xe72e0fb793cbbb04,
                    0x0096ca1001968a8d,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x01003d1bfde1255c,
                    0x882e53614bc4370e,
                    0x361d9850def24bcc,
                    0xc25c8e7c1fa23605,
                    0x37c6efc6037ab319,
                    0x018fd11232825425,
                ])),
                Felt::new(BigInteger384([
                    0x064a32d92dbf2f6b,
                    0xafc24c119857b99e,
                    0xe874e8a2af326753,
                    0x2f34de02aa07e7a5,
                    0x3a1d2214c97e3216,
                    0x012ae8eefaacd489,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2f5672821ccab04f,
                    0xc3eb620d8dfafb4a,
                    0x0ebe1a77abd3dc5f,
                    0xb57693d0f1805469,
                    0x61ae651047b717a0,
                    0x0009bce2c646e009,
                ])),
                Felt::new(BigInteger384([
                    0x20bd098e28e36d46,
                    0x90ba9cf7daa54f2f,
                    0xbd5d9328321b63e0,
                    0x17f387c703538530,
                    0xd9d9733c43d2ca3c,
                    0x011ab54799297c95,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xf9f8fe88e05f2dbf,
                    0xf28b263b1d4716c8,
                    0x30e28064c7aee33f,
                    0x9bf7dbfc0b2ceaa5,
                    0x9f6c09fa250754c5,
                    0x01814df5c72bf05c,
                ])),
                Felt::new(BigInteger384([
                    0xf7e8a4d294c7358d,
                    0xcdd8936e231eff99,
                    0x3114fa2b09b853f3,
                    0x40f4bfa1a96a127f,
                    0x50de62dc4d726cb7,
                    0x00eb5077ea342540,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2ec95e99c84778b7,
                    0x02acb459b1eb14f4,
                    0x36981c03714d0069,
                    0x65d3c81a66786edf,
                    0x742dfbb09aed4ed5,
                    0x00356ce46b43866d,
                ])),
                Felt::new(BigInteger384([
                    0xd52ccf2564d9e0e9,
                    0x395abcc0d798c946,
                    0x8f1b5ab72c863077,
                    0x18c1d61183df51d0,
                    0x171ca75ca445b2a1,
                    0x019ae05dcf67e734,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4a72700e39c7a45a,
                    0x102563e638209915,
                    0xced71be3493339e4,
                    0x916eef0799411a81,
                    0xea9654b3a4c3828c,
                    0x0028e43a5427ff60,
                ])),
                Felt::new(BigInteger384([
                    0x6bf081fe1bcd2914,
                    0x36c91e6965475f39,
                    0xc615714fde1b8280,
                    0xa580bfa58d884085,
                    0x3bfb115afab4793d,
                    0x0199263a5dbf7d16,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x1507ace1002ccf80,
                    0xb9e7b465de95a51b,
                    0x94145f61f95abfa5,
                    0x35d6bef71b0f3a6d,
                    0x1878e253d703d4ca,
                    0x00c2d23128c59061,
                ])),
                Felt::new(BigInteger384([
                    0x8026e473ce00949c,
                    0x3a9c1c096a5be51e,
                    0x085408e2131a145a,
                    0x1c5a530d9729cee7,
                    0x2276abcc7729ff89,
                    0x00a4c0ca086d9b40,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0814f74c6ef11115,
                    0xb1fcf6460feef933,
                    0x458efbdd7a073c0b,
                    0xd9ba59e068f81b97,
                    0xe83a1398b8c322d3,
                    0x01a6fa4290b15ab8,
                ])),
                Felt::new(BigInteger384([
                    0xade95d2db6263a67,
                    0x480d3494027138ae,
                    0x13b3eac277659394,
                    0x8309479e121f9945,
                    0x115f5a282b14e575,
                    0x003c9f31d3bcc25e,
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
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger384([
                    0x70733329590200fa,
                    0x840fa4ffe8a98115,
                    0x313551283042469f,
                    0xc6da48919248440d,
                    0x764a389c7f94b3cf,
                    0x01a27d45e13102bb,
                ])),
                Felt::new(BigInteger384([
                    0x998a5995fd5353d9,
                    0x8b841c7214e08375,
                    0xa641a679dc7a2acd,
                    0x29bb647ce2a4376d,
                    0x07f8480dd5deeb73,
                    0x003e28b3cce6d71f,
                ])),
                Felt::new(BigInteger384([
                    0x51980b228894568c,
                    0xaaaf9e1f0f2cddb1,
                    0x133f0dc9ec65fecd,
                    0x9b086e561bf313c3,
                    0x39b6a130041167e7,
                    0x017575115d9b7b99,
                ])),
                Felt::new(BigInteger384([
                    0x04f75065bc39e8ce,
                    0x046e048506ca55a2,
                    0x13b357889335f65f,
                    0x4550f78029c57c14,
                    0x85cf393109f623eb,
                    0x01490f4d1ab792f5,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb9361ef9f1675ba4,
                    0x4faa0b5c673e0971,
                    0x59140dad595485eb,
                    0xfe142feb629505d8,
                    0x6ec70351670072c5,
                    0x019c24aeea5b74d9,
                ])),
                Felt::new(BigInteger384([
                    0x8b02808f2f29a571,
                    0xb9c0a9fd7870326a,
                    0x710040dee3eace53,
                    0xd389f4555901c04f,
                    0x170bc0df6fc97a9c,
                    0x0078465d815f2e81,
                ])),
                Felt::new(BigInteger384([
                    0xd9b850acb06c72e2,
                    0xbd8d0d2e4bbce009,
                    0xd8d3c5b874066dbc,
                    0xbbec7af340755550,
                    0x137b25f67a5bc188,
                    0x00555548b5c5f010,
                ])),
                Felt::new(BigInteger384([
                    0x76b5dd2b1ed7b62e,
                    0x78aac731c97ed283,
                    0x9a945055fca5fc63,
                    0x2928526216356813,
                    0x58653c4f625f75c2,
                    0x00e694e09dc15643,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb0a5174cdb7659e9,
                    0x308a6ea73f7d04c1,
                    0xffed8bb367dd50ae,
                    0xc4a2828e67e849e8,
                    0xfa9fdfc54da10c2e,
                    0x0152105cd802899b,
                ])),
                Felt::new(BigInteger384([
                    0xddbcd0709b77ada7,
                    0x138e8e55da0b9272,
                    0x794d3c9cbb1158e1,
                    0x9abfea33b041125c,
                    0x1ea89261c4dccd28,
                    0x018f341ba3062c84,
                ])),
                Felt::new(BigInteger384([
                    0x8a04e6acc1ff8c1d,
                    0xfbde0d69faaf3b53,
                    0x6b812b88ca5a8aed,
                    0xab876e4c5b921798,
                    0x5664066b16a6bf6e,
                    0x007250efc9d51072,
                ])),
                Felt::new(BigInteger384([
                    0xcfd12fc40a19c606,
                    0xe98149fd579b0267,
                    0x23bbd97b8567e537,
                    0xf58fbdb48e67177f,
                    0xcf7b71841a831841,
                    0x0099fba6d6caf9c0,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe7aca492a7636439,
                    0x271bcac7a4c4d5c0,
                    0xb403e846ae5c0467,
                    0x44a58738af4f2c4b,
                    0x97214233e7f1f401,
                    0x01ac4ab199f573f9,
                ])),
                Felt::new(BigInteger384([
                    0x633846dbc8f5c356,
                    0x22aa262ba5a56b07,
                    0x4bd9dc4d46d5b536,
                    0x922517ca7c00d729,
                    0xbe14f6bd2df128df,
                    0x0040ea33f435e9d5,
                ])),
                Felt::new(BigInteger384([
                    0x740ed75e02d93982,
                    0xf5f4d9b8c8d951bb,
                    0x557bc2511b8fbf19,
                    0xde44c0360988fdca,
                    0x87bf6b4074a0cdb8,
                    0x002b05ea50ac7b89,
                ])),
                Felt::new(BigInteger384([
                    0xd6e2749e9b694e09,
                    0x4740fe3679218523,
                    0x9f8e633ee6b7c72f,
                    0xefa5d2dc7cea98fa,
                    0xb3368c367fdd7dd4,
                    0x013d4bfc6bdbf080,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x09a419cc3037463f,
                    0x7c31b8ebe4cda4e4,
                    0x338ab10ea4d0c251,
                    0x578dcb2609f8da55,
                    0xe1247e33e76312aa,
                    0x00aa4cd467545bbb,
                ])),
                Felt::new(BigInteger384([
                    0x1e9a4a7227df65d3,
                    0x00ad2bf47682da81,
                    0x34fd0787eaef5588,
                    0xdc4a9c886f63a894,
                    0x66ee40c566481dbb,
                    0x01a280b7fc795ede,
                ])),
                Felt::new(BigInteger384([
                    0x632bed9221a48b09,
                    0xfcd02155d70c00d2,
                    0xf411387ef81c595b,
                    0xc76a9ba0109a7cea,
                    0x05d17868deeecec4,
                    0x004d15e43fbc4bd3,
                ])),
                Felt::new(BigInteger384([
                    0xa71c2d3769fdf575,
                    0x1d21def96afac4ef,
                    0x8e22e18a2e882a0f,
                    0x10c5ca351b76cf10,
                    0xca8be301f524bd44,
                    0x0142fbac1a1a5c69,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x7f0e3bfd9896cd83,
                    0xc3c54d8b62d2a390,
                    0x1567a93549713f48,
                    0x4546552b8a8a9a4a,
                    0x4f3d23d7c2f55661,
                    0x009c4493fab7b4b2,
                ])),
                Felt::new(BigInteger384([
                    0xa58073c3bc9f9bc4,
                    0x10338ec7a5d60cff,
                    0xa955acc80fab617d,
                    0x90be2cd4149198f5,
                    0xe5b691a47f8770d9,
                    0x00a4546edaba2c15,
                ])),
                Felt::new(BigInteger384([
                    0x62814680e04c055d,
                    0xbcc4083d4c4bf917,
                    0x50c8864853d80af3,
                    0x91cdaaff648206c8,
                    0xce79a3107cb46d23,
                    0x0088fa6bc22a77c9,
                ])),
                Felt::new(BigInteger384([
                    0xb8006dfd63107ea5,
                    0x16a585ded42cb83a,
                    0x20106bf4b63b0ad8,
                    0x04ef5d81b4e3f87d,
                    0x5403fd7fc319083f,
                    0x00f8db936d1d1ab1,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xc2db01acfdf7099a,
                0x5a729515d5027380,
                0x44c3e11d2e266e71,
                0xd5bf8b915feae472,
                0xc3cd6d3851ca663b,
                0x014321a35a0b692c,
            ]))],
            [Felt::new(BigInteger384([
                0x73328f14332fa5e4,
                0xdae5d0ef545df4b7,
                0x4bff326ab478b6d9,
                0x69ff669e33d0997d,
                0x3a544451fa36acda,
                0x0136f06b2606d64d,
            ]))],
            [Felt::new(BigInteger384([
                0x3d2c3a45dca4f39e,
                0x0ebb2f0ed44c9371,
                0x9228efc15858b824,
                0x661d1b180ae922d7,
                0x3bb4df7e9eabdd80,
                0x002e0cb2eeb33b7e,
            ]))],
            [Felt::new(BigInteger384([
                0x8241aff52ba054c6,
                0x20e5422eb41bf0ab,
                0xff9f1ec3d41b6b20,
                0xd76e928bc8b50a1b,
                0xaba90c1a60579bf4,
                0x010c7fbb156a17c3,
            ]))],
            [Felt::new(BigInteger384([
                0x5c36e14a404a3832,
                0xe6c7028fcefe54c4,
                0x5ed0734a70833cd5,
                0xc56cbfed0ae7add5,
                0x471b68afb33cafab,
                0x0069e442b15a3bc9,
            ]))],
            [Felt::new(BigInteger384([
                0x8d58e4e004e7dc5b,
                0xc88a4cd5ef644ad9,
                0x19fe67bf793415d4,
                0x5a1769ae6cde6b91,
                0xdd4cb75bcd9801e4,
                0x013e10ad033ede95,
            ]))],
            [Felt::new(BigInteger384([
                0x51bc0a29b4488a71,
                0x53d986390ccc3147,
                0x7893a3933162af4a,
                0x55906dfe5e685649,
                0xe7842bba9311fe0d,
                0x00f31d4383f91d3f,
            ]))],
            [Felt::new(BigInteger384([
                0xe730286b6577ed57,
                0x4c0aebd5e94889af,
                0xf5edaf0c9f5d3ee7,
                0x1624b1880f823948,
                0x30994011e4fc7b2d,
                0x00ab2c9def8b20e5,
            ]))],
            [Felt::new(BigInteger384([
                0x50e9e7d8af122786,
                0x1a5bb41be6d33b23,
                0x7db90e4406cc9e19,
                0x1b059dd9cd16e0e7,
                0x0750367df5e7c56f,
                0x00ba77d48a33fa9f,
            ]))],
            [Felt::new(BigInteger384([
                0x61a023f4bbcee1c6,
                0x2de68019b5f335df,
                0x6e5e48b85904c569,
                0x39d8b9e66b0a49e7,
                0xdde8a420eb9e7cf0,
                0x0006aeefb9be745e,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
