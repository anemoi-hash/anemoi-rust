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
    use super::super::BigInteger384;
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
            vec![Felt::new(BigInteger384([
                0xca9c794f2931043b,
                0x149f8e9d5f93fa4d,
                0x6827d62743d1b065,
                0x110de602ed634026,
                0x2b5b470a4b57edb3,
                0x153bc67ea4dd3947,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0xd09c2d0b8ed35e40,
                    0xd7a774dfc5d69c8e,
                    0xde74733e138c05ec,
                    0x0032cc5fa7596f91,
                    0xe0c537ce3f0b7067,
                    0x0a59ab5ce650887b,
                ])),
                Felt::new(BigInteger384([
                    0xe7601d3618b11558,
                    0x2aa2069a63c920e1,
                    0x493f88da902e8784,
                    0xfca7044bfd04404b,
                    0x9c8da1e49efbeb9c,
                    0x01462be5f0d89cd5,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x36befd1d6e265aa3,
                    0xed600f53091cd0b5,
                    0x7ac29fe59b8f0f5d,
                    0x7eb9616409cdeb74,
                    0xb64b64a44411c0be,
                    0x16b033cbb7bdd8a3,
                ])),
                Felt::new(BigInteger384([
                    0x9f28809f2daa381f,
                    0x2c5ad84071dfb79a,
                    0x770e9b93abece01e,
                    0x87d0107861c1aae3,
                    0x0731ef31305e2bf2,
                    0x15a0ca405856f73e,
                ])),
                Felt::new(BigInteger384([
                    0x569f380e2bed59e6,
                    0x781769bc1cbebf51,
                    0xfd473961a17053bf,
                    0x8ddd205b2007a8dc,
                    0x0c0bf0adafbef42c,
                    0x134c693723096dbc,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x2d5b4ed9fcd46ded,
                    0x4a21b694403f8bc8,
                    0x71e4244021573191,
                    0x01e2c9fba0932c52,
                    0x8f020981c99c5551,
                    0x1301d7196c4a823a,
                ])),
                Felt::new(BigInteger384([
                    0x98490fc906eedb4c,
                    0xb42eca1a6b2064ea,
                    0x50005495fd9b4aeb,
                    0xb6504044c9fd38b2,
                    0x39311514fb6b6ee7,
                    0x01250d3cb7e041b4,
                ])),
                Felt::new(BigInteger384([
                    0xb897bffba479c406,
                    0xb0a4a846afd8c738,
                    0xa4dc9b8e33c3990c,
                    0xe707cc8d686079ef,
                    0xf7e1f7f22b581df7,
                    0x0a4dd533eada0ea0,
                ])),
                Felt::new(BigInteger384([
                    0xfa503b3a22fef5ee,
                    0xed159440b6208c81,
                    0xe622632ceb6e8852,
                    0xc56c14d07681f1d8,
                    0x6792904e2102ba29,
                    0x00951ac90a57450e,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x8b5bba28244853f7,
                    0xf2e02f0643ccbacb,
                    0x7b51654e155bc770,
                    0x05dd47eed794f239,
                    0x8b611ce476127548,
                    0x11d0ffdf031f1e56,
                ])),
                Felt::new(BigInteger384([
                    0x47e9aaf5942c5a18,
                    0x0936903859c45451,
                    0xc63d97f1a1db82e1,
                    0x2a6caaf8d79d9dd4,
                    0xfd0f8463e6433e9f,
                    0x149fd69d268aab00,
                ])),
                Felt::new(BigInteger384([
                    0xda065e899d6bba24,
                    0x64cf47e3cf845c64,
                    0x66bb390e9badb587,
                    0xa195868c6f24519e,
                    0x51d5c2b495d850be,
                    0x154227c460c14de5,
                ])),
                Felt::new(BigInteger384([
                    0xb1d8a37f20ffe08a,
                    0x00fb24a25ad2d2b8,
                    0x23cd576cb90f1cd2,
                    0x964bf5ad260e83c0,
                    0x8b73e0e2b92b971d,
                    0x119dc52120988246,
                ])),
                Felt::new(BigInteger384([
                    0xc2b711424eb9362f,
                    0xcdca929c769baf24,
                    0xc1d2f800ad467fca,
                    0x6c4831bf852f9207,
                    0x3029e1e493827552,
                    0x18e810779652f4d1,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd75d03a3f47ce294,
                    0x60109cdec7c44693,
                    0xd042888e31dfd9d1,
                    0x68ac9993babf2c2e,
                    0x0f29fa64dfb6a6e2,
                    0x0754cdf7dc00ef7f,
                ])),
                Felt::new(BigInteger384([
                    0x4a49f01a8997a893,
                    0x1ff30e0ac4e9fded,
                    0x0c0b6b66e4571674,
                    0xb1909f87fb72af3d,
                    0x8aeb25c712c488f0,
                    0x09b538879cdc18e8,
                ])),
                Felt::new(BigInteger384([
                    0x7b9711f6d2e81d3f,
                    0xa04816f0cbd18548,
                    0xa07c37c7a8e40666,
                    0xfadd35840b57b28f,
                    0x3706f89781c05006,
                    0x027b1cb776170226,
                ])),
                Felt::new(BigInteger384([
                    0xc375117e554a17a9,
                    0x6f2297032cccb00c,
                    0xf4ace8dbd4ef61ce,
                    0x4faa3e24b54d7d75,
                    0xaac5d4902fbd5560,
                    0x0c80050b9e01c2f6,
                ])),
                Felt::new(BigInteger384([
                    0xb0f810037d60c741,
                    0xb4a1b49f84c413c9,
                    0x8f98ca5bcda0c02f,
                    0xf22bce85a8c4f08f,
                    0x684568a7c76b2cb0,
                    0x01b6f5508a172911,
                ])),
                Felt::new(BigInteger384([
                    0x213b7bfd8547c34d,
                    0xe05ab19fbf283fd1,
                    0x01bc6969e6105dbd,
                    0x80545f5d849bcbf1,
                    0x351125a1cc67004d,
                    0x17455858512eb142,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x80a45c0c08b44057,
                0x55f5efa62dad5146,
                0x4ce05f432bfd26f0,
                0x8e7cf30a43e3b4d0,
                0x495b59d0fd722163,
                0x0f9cf36953cde49d,
            ]))],
            [Felt::new(BigInteger384([
                0xae5739aac179e0cc,
                0x62dc38688f68ba9f,
                0x5006ed82fc1e32fd,
                0x814770e735181f1a,
                0x543f5618ef3fb339,
                0x00b3d63ccb2d6457,
            ]))],
            [Felt::new(BigInteger384([
                0x9d2ab2aefd1c9b3a,
                0xbcf1285a0c42600e,
                0xca4c8ec8e97d55cd,
                0x67628935a1dec831,
                0xff4a4eac3b861f08,
                0x0911627afd8cac53,
            ]))],
            [Felt::new(BigInteger384([
                0x44b357c150105dc7,
                0xea5b487b450f6d9d,
                0x4dc851ff8d81fa5c,
                0x8b0a454f7a594be1,
                0x91fce2e5b7042811,
                0x1586f83d7f66a2c0,
            ]))],
            [Felt::new(BigInteger384([
                0xe490d5349459f925,
                0x0c6c9dc334bce3fc,
                0x30c486dc8757f95f,
                0x6f9585dafa2f74d0,
                0x2635e42b642e63b7,
                0x15eea0de6b343329,
            ]))],
            [Felt::new(BigInteger384([
                0x1360caeb8e9b3f29,
                0x3faaf300d6042c73,
                0xeeb77f7a615eb968,
                0xe3293ba8342254e7,
                0x6ea18113a8e674ae,
                0x1373880ae5b28651,
            ]))],
            [Felt::new(BigInteger384([
                0x6e2df15094af9417,
                0xafbb1bcf7e9ee44a,
                0x332c3f14cea30320,
                0x808a7ad2a4a1f9ae,
                0xc523126a3c4c8e62,
                0x17e6106675a3a26c,
            ]))],
            [Felt::new(BigInteger384([
                0x78fe4e534da147f6,
                0x23ca869490ab1a34,
                0x8ee4bd1997d566d9,
                0x478f10042908b07a,
                0x4c739d37b9c6cae3,
                0x18efaf2b8d56fa42,
            ]))],
            [Felt::new(BigInteger384([
                0x298a7e4f39833922,
                0x6d0657e2ae2f5aae,
                0xae9293cf5fad2ace,
                0x78d50988947ae709,
                0x9a7975acedc07aea,
                0x086387eee31ce236,
            ]))],
            [Felt::new(BigInteger384([
                0x986caa805a2b5d3f,
                0x41ec87b595284ca1,
                0x092c3a9c41f37358,
                0x6ba73ba44e5682a6,
                0xd97c4658517a43bc,
                0x15aa78ac714e5dbd,
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
                Felt::new(BigInteger384([
                    0xd06613b905cbbddb,
                    0x43fe6618eaabca48,
                    0xf4338d7eacf7d183,
                    0x98b02f887aa6908f,
                    0x550df4430e607710,
                    0x040dc41fb6872433,
                ])),
                Felt::new(BigInteger384([
                    0xa6e529f0c16aba4f,
                    0x99745a8e066c3db5,
                    0x50be327c7fe09d3f,
                    0x63563d4532fb0e61,
                    0x5b6fc28c4ee25ff8,
                    0x01405ef0e9a52c50,
                ])),
                Felt::new(BigInteger384([
                    0x88f91604459ee3d1,
                    0x07d15cd2ac47144e,
                    0x8c0b5e997d99be19,
                    0x4ee31c657a103628,
                    0xa932e2b6a548e2cb,
                    0x1996d8a290fdf3fd,
                ])),
                Felt::new(BigInteger384([
                    0xf0442ac34928d31f,
                    0x5799fc7a84caa7e0,
                    0x7b4a1bf630b19bfe,
                    0x7e1ecd9cc56141ae,
                    0x5784e71abff934f5,
                    0x0395dfa889478f0a,
                ])),
                Felt::new(BigInteger384([
                    0xea21911176e4e59a,
                    0x8658b5259f197eca,
                    0x4f2b356cb60d4ec3,
                    0x082d7a141bb524d3,
                    0x3a4b4962605531a1,
                    0x135fe87d1ca92cfa,
                ])),
                Felt::new(BigInteger384([
                    0x810d5ab2d89d9fba,
                    0xba92b9ad34810637,
                    0x945b16bbe0a28e4a,
                    0xdb5cab988d7cf3d6,
                    0x88010fe4c6ffea6e,
                    0x09418632577222a2,
                ])),
                Felt::new(BigInteger384([
                    0xa520a3e88554b89e,
                    0x3466d31fc13b9164,
                    0xb869c6d14080ed76,
                    0xa209544a4f7fd6ed,
                    0x80f477e6f58c6659,
                    0x180a3bf5f034c174,
                ])),
                Felt::new(BigInteger384([
                    0x6593bc74dc660960,
                    0xe1a375e99570b51f,
                    0x3c3c50a07ddc89ef,
                    0xd221d1b1a1348346,
                    0xb0bc763edc027e25,
                    0x1132f930a2fce453,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xde5650ec8c38c880,
                    0x25824a5cf6a97d53,
                    0x26c97fc458fe4de5,
                    0x292cd58ccfec2469,
                    0x2c34e50f638af559,
                    0x156926a3910c8772,
                ])),
                Felt::new(BigInteger384([
                    0x6e45c80cc5f180ee,
                    0x5b3741f113faecd0,
                    0x3d8174053260ac3b,
                    0xafab4c2d93215806,
                    0xee9fe41ca3d75b21,
                    0x11965ab69f0c1116,
                ])),
                Felt::new(BigInteger384([
                    0xab0380a30d4251a6,
                    0x8f9dbb87d0436564,
                    0x83bddbb84539302a,
                    0xedbc44e0deb00255,
                    0xdccbf97df585b758,
                    0x08c5720ca0dbec8a,
                ])),
                Felt::new(BigInteger384([
                    0xec5abdf6203f0c07,
                    0x619dfc82fa9934eb,
                    0xb73643456db1de75,
                    0x0205ede0daf3b531,
                    0x025f24bb89ca00da,
                    0x0c9df1bc61d5ba22,
                ])),
                Felt::new(BigInteger384([
                    0x3810093bbff6e22f,
                    0x3893c90b43a6777a,
                    0xa556366805994005,
                    0x8ecac02082305e1c,
                    0x31735031e48ca88a,
                    0x18faf70f1d2f5562,
                ])),
                Felt::new(BigInteger384([
                    0x7cebfb63b7ad5fe2,
                    0xb675e3e7c1f7c7a9,
                    0x908b3940f4f81833,
                    0x7c4c204db718b86f,
                    0xa547dc76f42c6890,
                    0x151927efd6562807,
                ])),
                Felt::new(BigInteger384([
                    0x95d39c4a3206e682,
                    0x53bb66e067ab38b3,
                    0xddf0863e0bb21d11,
                    0xb82a47165ce9c5a3,
                    0xbc7ee6989cbb900a,
                    0x08c2ff9f49bf84a7,
                ])),
                Felt::new(BigInteger384([
                    0xf393581d58dd6743,
                    0x8d13f1f065e23af2,
                    0x99bff500b3ccd2ca,
                    0x30c92ac53020344e,
                    0x05b0bc951afc6f76,
                    0x12a8c7a00fb314c7,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xdef31eef0b2287e8,
                    0x9faa50dd2e6dc194,
                    0x94df3cf3bf317cf0,
                    0x5b807349326002ca,
                    0xe67cf16c23d38d42,
                    0x16d729d09241e152,
                ])),
                Felt::new(BigInteger384([
                    0x79ae774d8aaf93a5,
                    0x3a02f18ba9181043,
                    0x8aff55127b175b53,
                    0x0e7d82c1c00ff391,
                    0xada135f01c0c39b0,
                    0x0b1fd9d29328fcb9,
                ])),
                Felt::new(BigInteger384([
                    0xaefe676ac8bf9536,
                    0x0ac3c05f6d24fa5e,
                    0x7ed4d024773801dc,
                    0x6b33f42e1b266803,
                    0x3da0b758ee822dc7,
                    0x1029d5177248335c,
                ])),
                Felt::new(BigInteger384([
                    0xf127227bfd3e726f,
                    0x886418cc9765d487,
                    0xacd10c4d3fdcd051,
                    0xb3704a902fca7c14,
                    0x665c7df66bdadefc,
                    0x0c4da714a738db8c,
                ])),
                Felt::new(BigInteger384([
                    0xee9b3c9d87094742,
                    0x6dc8ba0172dab821,
                    0x4736d8a095708edb,
                    0x278576e3673fa6e3,
                    0x68e5db8d395ab17e,
                    0x198823283cfe0f11,
                ])),
                Felt::new(BigInteger384([
                    0xd4e4ad3afd3e729d,
                    0x6caaff404dcb937d,
                    0xc8aca3bd21320877,
                    0xe40836dab591173e,
                    0xadb4935d190ef84b,
                    0x0e505b90041f48f7,
                ])),
                Felt::new(BigInteger384([
                    0x387435eb54c7de37,
                    0x090cb2699f79986a,
                    0x7e0ab868198ecd86,
                    0x5e7b9c8d306d8747,
                    0xec12a87701e5c803,
                    0x07ed49355a3e7a03,
                ])),
                Felt::new(BigInteger384([
                    0x099b304e7ff71a61,
                    0x409f73f1fe1ad9e0,
                    0x632d21f63cc2e5d1,
                    0x5cc2bd73f4826d53,
                    0xb537ad36395d95d1,
                    0x061a4ca6b909a1b6,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x18d4f31f5931fe08,
                    0xe35598f0d6edb9a0,
                    0x4ddfc08ab85eca80,
                    0x051435257418d1ca,
                    0x8146e83d33e04f66,
                    0x05e54e49ecdaae99,
                ])),
                Felt::new(BigInteger384([
                    0x600735066a7a18ab,
                    0x605068ff04bc5aa2,
                    0x0bf0045edb4b33ba,
                    0x5cc79035d7e20e40,
                    0x75854d3700704cbf,
                    0x0ce59396ee1f1ba1,
                ])),
                Felt::new(BigInteger384([
                    0x0739cc38e21c6ee1,
                    0x997852dfdf4af917,
                    0x1c919ba89c452864,
                    0x53967f4047b2fae7,
                    0x4f70bebb74b992c4,
                    0x15e289ab27e3b106,
                ])),
                Felt::new(BigInteger384([
                    0x763097528c3c882b,
                    0xf2c7827114e66ec6,
                    0x463d680ae8d2ecc3,
                    0xd8d33e48b0cf61b5,
                    0xb9d5343bcb747fa1,
                    0x14e1bc8e4693fa9c,
                ])),
                Felt::new(BigInteger384([
                    0xa64fcb0095e0b281,
                    0xd1e837c1dbe3456d,
                    0x60ba9699eb835c6d,
                    0x9dfc7e72f472839f,
                    0x42fbee297ad63b4d,
                    0x0dd9aeabf7189fd9,
                ])),
                Felt::new(BigInteger384([
                    0x952a5ed3eccc0167,
                    0x796f1fcc3061337b,
                    0x5a8a1835469dd3e9,
                    0x987a045a5f1e1c99,
                    0xa2481e27bc3ef6d7,
                    0x119457dba90aa59e,
                ])),
                Felt::new(BigInteger384([
                    0x072cd9f181908c6d,
                    0x9fa4852350230be1,
                    0x1b7079d1ca6693c8,
                    0xdaa3a45929b27451,
                    0x87d5ab6bf89fbc22,
                    0x02420f0eaadcf62a,
                ])),
                Felt::new(BigInteger384([
                    0xb831e6ff9a190091,
                    0xd7d332aadad68292,
                    0x79cff9837e043f76,
                    0x2857139912e60d6e,
                    0xf2aef45b374f7e74,
                    0x065d9502235fbf1a,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x6d81d9a0b0cb502d,
                    0x40c760db0012f07e,
                    0x09ff2f30d1660a3c,
                    0x15e21d2b0cdade26,
                    0x3d08332e481ea998,
                    0x0846f5f8436495d8,
                ])),
                Felt::new(BigInteger384([
                    0x285e8f0ad616ed2f,
                    0x6eb985de5eb51d5f,
                    0x8395541075ce839e,
                    0x4c67d747f193e31a,
                    0xa8a66a66a0ed5d3f,
                    0x14eb90ae97e15c5b,
                ])),
                Felt::new(BigInteger384([
                    0x9ea9f184530a5b29,
                    0x0cd5a2f880db11e6,
                    0x845cd8bbd74d8d5d,
                    0xec800474df86a1cc,
                    0xef8dde087d118323,
                    0x0b593c0f269d914e,
                ])),
                Felt::new(BigInteger384([
                    0x48ba95662bddae72,
                    0x68fe0ed0e5f4b58d,
                    0xb18717e4d0b6dc74,
                    0x378d953389ba220d,
                    0xd1541449177a8237,
                    0x0a83bb15213a13be,
                ])),
                Felt::new(BigInteger384([
                    0x0ec6b5c8f20e64e5,
                    0x344fcfeb630a8ee9,
                    0x3268dcab4de6d95b,
                    0x883eddf564cc69eb,
                    0xa3fd9303816c3c89,
                    0x0564d51d9e883817,
                ])),
                Felt::new(BigInteger384([
                    0x3f8334b3540f505a,
                    0xde0fc0193450367b,
                    0x2efaed7f3d99b88e,
                    0x2f33399f102eb653,
                    0x794d2b5734a5d289,
                    0x10e4793b31a33205,
                ])),
                Felt::new(BigInteger384([
                    0x85e67cd6d3a5bdc0,
                    0x5053ee553ecd098b,
                    0x55d2c0f401a8d35c,
                    0xceb58e0d5bba8da6,
                    0xccff2aea472fdba4,
                    0x05cbbc6a792b7d0b,
                ])),
                Felt::new(BigInteger384([
                    0x71ecd27189db4ef8,
                    0xc0a54f476d646431,
                    0xdd267e2a145c289b,
                    0xb9fb38d60c4627f7,
                    0xce8a8436b7baa558,
                    0x0a33afb458f2868e,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xc806fb2d9e84e009,
                    0x3f4f5922da96e764,
                    0xb768f30ded38d5ac,
                    0xb151346d5eff1b41,
                    0xe18075eb03c4de04,
                    0x129d3a2e0061d23b,
                ])),
                Felt::new(BigInteger384([
                    0x93cad1e533e11a4f,
                    0x14db382d42b6fdb1,
                    0x00abd3351632f5bf,
                    0x964d8122c254fb8d,
                    0x1382dd17375b037e,
                    0x0b10ee4dd6a0d537,
                ])),
                Felt::new(BigInteger384([
                    0x0f319d4b48fbd7a2,
                    0xb4be535b19bb8cba,
                    0xe59e64426bc496ca,
                    0x5da9b67ca6b288df,
                    0xd2c231c2cf689352,
                    0x0d571cd5d06662e3,
                ])),
                Felt::new(BigInteger384([
                    0xeaa89b8d2fbbaad7,
                    0xd186383dd4b9f631,
                    0x2dbb947a806672bd,
                    0xfcf35aa4da57d076,
                    0xe842a49d0da49316,
                    0x0b387a54e268e753,
                ])),
                Felt::new(BigInteger384([
                    0x4ea0a7fa4ce15903,
                    0xe179306da3eee59d,
                    0x88cb6869e37356f5,
                    0xaed2d56cbcc3d902,
                    0x0cb278620b1c7e61,
                    0x0ae2316a74705434,
                ])),
                Felt::new(BigInteger384([
                    0x1e2107399ba081a1,
                    0xc860d060f0840eae,
                    0x5204ddedd39f976b,
                    0xa891ff5d7150b0f5,
                    0x9621d493058740dc,
                    0x0532ce7dcaab47f6,
                ])),
                Felt::new(BigInteger384([
                    0xba30c340abc56edc,
                    0x0f7eb0281a6027a9,
                    0x1933c812ca2628f6,
                    0xb9ccafe6ce0287fc,
                    0x7ead8ffe161ff34d,
                    0x10dc9c381641a322,
                ])),
                Felt::new(BigInteger384([
                    0xe41044a2425555a2,
                    0xfdefd50d11b3a7f7,
                    0xb3bb7814633e689d,
                    0x32894b7932eab4c4,
                    0xf11ddc21c51c4da1,
                    0x0e5b2642c4118d82,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0x39c11c6b2d01e810,
                    0x101732d6a26cdd7b,
                    0x4bdf8a7927ac15a0,
                    0x39b42828a0f08c18,
                    0x239f410384f96951,
                    0x0cb19b09cdba5901,
                ])),
                Felt::new(BigInteger384([
                    0x32f5a267b189fda6,
                    0xe442491bc0ee45fc,
                    0xd74a6b8ac7946d94,
                    0x72f12cba3f946519,
                    0x902cff11073a54f4,
                    0x16e21c8b903e9c0e,
                ])),
                Felt::new(BigInteger384([
                    0xa61764300a07be28,
                    0x9ef2211ced2a5c9d,
                    0xa2b449f438c8219c,
                    0xb909bb44be21d573,
                    0x6d8b304fc9abbe14,
                    0x105868fec78b40b8,
                ])),
                Felt::new(BigInteger384([
                    0x1cfbe8c5e57c5a98,
                    0xd78104b41628e2f3,
                    0xb724578f1d3c9575,
                    0x0385243c55094211,
                    0x37dc9561bf5fd97c,
                    0x15716692f6f08bda,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x0f8f7d2db9e321eb,
                    0xd00aa5dcf569491a,
                    0x0a78bc635d95dbf0,
                    0x6aa874a4aee11706,
                    0x93c3d70d2526d036,
                    0x06caec5450d2acb9,
                ])),
                Felt::new(BigInteger384([
                    0xd728252b3e145c06,
                    0xc6c35e3a3d0337de,
                    0x87b063bce81d4aba,
                    0x82b97b41cecd8270,
                    0x01a586557f6d89f0,
                    0x0e1a0c545651aa57,
                ])),
                Felt::new(BigInteger384([
                    0x91c1a0469476bd59,
                    0x75273468e6bb20be,
                    0xebe26e368f75ad04,
                    0x14660bbc2eda0990,
                    0x82c918b06e3c9cba,
                    0x03feac2c15eda240,
                ])),
                Felt::new(BigInteger384([
                    0x7973af695dbaf007,
                    0xc042569f8dca0b75,
                    0xe2a375db76bab431,
                    0x358a1a102570acf5,
                    0xfd0ed6d5c318843c,
                    0x0447d0c186704c90,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x88fb1df8fcf47f7f,
                    0x74a2809f4096e1db,
                    0x73bee67540ff2dff,
                    0xfd2e5b4dc5fbc18a,
                    0xd72acd5a26916cae,
                    0x0e9fa60af88a5f7f,
                ])),
                Felt::new(BigInteger384([
                    0x81946ded4e368b77,
                    0xaffb66106bd64f41,
                    0xa08178d35e7634de,
                    0xa08cda7b6cbae3ab,
                    0x63d60dbfd3a4c844,
                    0x136fe498f1b161a5,
                ])),
                Felt::new(BigInteger384([
                    0xa361977743df26a5,
                    0xc253b24d16ec73ef,
                    0xfefc0d4bce1b71d6,
                    0x8856b892c77412c6,
                    0xc7c3f524a36e8f73,
                    0x02fede33fa13ed46,
                ])),
                Felt::new(BigInteger384([
                    0xb59c8f110d630ace,
                    0xa4c5ea9a57f2cf4b,
                    0x430ba32018787734,
                    0xfcf221707ee74bf1,
                    0xf7c343c85be582a4,
                    0x02f15ee70185bd15,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x83598e0a0885a7c1,
                    0x5f061eb40158a802,
                    0x0e422dbe36b22755,
                    0x7d69feafe9076573,
                    0xc671e4823598a9cc,
                    0x18b960a631295e2d,
                ])),
                Felt::new(BigInteger384([
                    0x82eb10695fe8bdc9,
                    0x3f455b0bcafd3d42,
                    0x73d5042721c6c40e,
                    0x3e17913ed1799cf3,
                    0xd7d66e51099753a9,
                    0x1765edd03bd8a2bd,
                ])),
                Felt::new(BigInteger384([
                    0xc6f42db079e0644c,
                    0xd5243f5e3941eb89,
                    0xa843f820f734783d,
                    0xda25a1907ce7f4a7,
                    0x50b59acf45ac1d5d,
                    0x0965469f8a0f8d87,
                ])),
                Felt::new(BigInteger384([
                    0xe0a69b247cb0fdf3,
                    0x2790bfca2fbd5d32,
                    0x65bfd1e83c74f544,
                    0x373ef9dc36ab65fa,
                    0x652b54d91134b10f,
                    0x0d906989ca4d9c6f,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x37466d22cbbceaaf,
                    0xc8dd0cdf224b9d8a,
                    0xf72e2acdee936ebf,
                    0xbcdd88bb404510cf,
                    0x3903ea1e5bd246db,
                    0x08b938835bee613c,
                ])),
                Felt::new(BigInteger384([
                    0x1670daeacee7a7e2,
                    0x87eb1a101739f1e5,
                    0xf1df6d087fc552c1,
                    0xd1b79d3cb8d1705c,
                    0xb2c9755342177c71,
                    0x0a1eb08a57037879,
                ])),
                Felt::new(BigInteger384([
                    0x3673758e2615052f,
                    0x123bf8576e932914,
                    0xe3d287392575ed63,
                    0x6b6b31a7fb591080,
                    0xbb66142b917d8128,
                    0x1705acee39269c2f,
                ])),
                Felt::new(BigInteger384([
                    0xe2dc15ae72f89e3b,
                    0xb0910469d082c28b,
                    0x75f85f5c42e70645,
                    0x67479acfcdda2bfe,
                    0x0c9e1a03adfb821d,
                    0x05fa6a6a140c9fd8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x95e604fc139defb0,
                    0xce51565b0367e8e9,
                    0x26a46deda567ac65,
                    0x8e2c779270fa6cfb,
                    0xc67a92f142867558,
                    0x1127274429f3a0ce,
                ])),
                Felt::new(BigInteger384([
                    0x1194edd04d69b0bf,
                    0xf8a95882a435062c,
                    0xdd6ecba6604689dc,
                    0xc547e450d2518057,
                    0xe13b20c41b29c0ef,
                    0x0e5a80c1c5b96b76,
                ])),
                Felt::new(BigInteger384([
                    0xfb7f1ae5c7ab89f4,
                    0xb843907f37b1d9b4,
                    0x8034c726c6580cc3,
                    0x615ca6372ad618e5,
                    0x6ecea8195deb02f7,
                    0x0b43f7c92e0f4351,
                ])),
                Felt::new(BigInteger384([
                    0x6db97efed4faba62,
                    0x370edabfd74572da,
                    0x467fc3bb9d07f4f2,
                    0x26a3f7bee6311e0d,
                    0xb1f71614b8252fdb,
                    0x0c9364a11084e63b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x70ac797606f04957,
                    0x940e5e05673f296b,
                    0xbc7c77b867c7a400,
                    0x9c0c988dd1bc1551,
                    0x0a50cd644da90a85,
                    0x03cb5142588b51f0,
                ])),
                Felt::new(BigInteger384([
                    0x2bdd5fcfddfc7a0f,
                    0x0431342f49461a4f,
                    0x694eee3c8bd1e97e,
                    0xc231f5d128ce5fe4,
                    0xde5e2215ec09ace1,
                    0x0ab2bc3039af51df,
                ])),
                Felt::new(BigInteger384([
                    0x5a8118285f68c032,
                    0xe15a98d80235823d,
                    0x35e3639e2d2e0c33,
                    0x02bacdcaa45fcd0b,
                    0x8254b9566fa99fcd,
                    0x092489a049906024,
                ])),
                Felt::new(BigInteger384([
                    0x70d6d870159294ea,
                    0x77e1b2f793110577,
                    0xd25ed3e1f8a2bdc4,
                    0x937c6ab74c0319e8,
                    0x1d1fb670c86f85aa,
                    0x1935b85854ee98d8,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x465e238907c04d75,
                    0x16f23ddb76d5849b,
                    0xf776a6fe35395436,
                    0x803d9651b3f2d5ac,
                    0xd348c408e5761375,
                    0x02257bf8eb640331,
                ])),
                Felt::new(BigInteger384([
                    0x7390705c7d582b0f,
                    0x2bd4cffd68a01d72,
                    0xa5d36d5f1770c125,
                    0x81e0878c21229953,
                    0x4cec1273cb7f7e48,
                    0x082fcbc65c1b7b0e,
                ])),
                Felt::new(BigInteger384([
                    0x3c25773d5b9e6470,
                    0xd023841a6b08626b,
                    0x1b4b5dbfa6a3da99,
                    0x70bb3c2d724a7811,
                    0xd1abff38f343996b,
                    0x051f576c02fb161b,
                ])),
                Felt::new(BigInteger384([
                    0x8e558134276968e7,
                    0x77e60526f6e73be3,
                    0xd5e7f6fe9cd9f1c0,
                    0xf5bbf6fd7b218567,
                    0x1a662872e60defcf,
                    0x11c3e0957a89911c,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x3a7f7190728da9be,
                    0x26e93bb136b51653,
                    0xcaa9bfc294191514,
                    0x6a75cd95f09e178c,
                    0x351ab7feaab3689a,
                    0x1087c9826268b059,
                ])),
                Felt::new(BigInteger384([
                    0x27c6b6285117b3b8,
                    0x9f6cd27d1773c384,
                    0x0bb1787ae015eb05,
                    0xde15445fab04ad5e,
                    0x0702522ba781041b,
                    0x0b4daf7e3c4dfa8f,
                ])),
                Felt::new(BigInteger384([
                    0x79178f3faa9ca1dd,
                    0x352a24d2dbac0868,
                    0x4f5a6bdff6bd0e77,
                    0x14bf2a3f265ddb3e,
                    0x610dff8c5cc8a3d5,
                    0x064f27346b547c05,
                ])),
                Felt::new(BigInteger384([
                    0xeea7ed6e2e74bc94,
                    0x9ea948b812bf068e,
                    0xa60f6c7efcd31322,
                    0x81569921297950d7,
                    0x30b85c9a2dcbb7f9,
                    0x110d5dee82f570e4,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6f312dd77a30aec0,
                    0xc742176026845bb1,
                    0x9fa0f29ad3b2da5e,
                    0x32297030beff4044,
                    0x52ed6193255a8e4e,
                    0x02cc4c7e12935d64,
                ])),
                Felt::new(BigInteger384([
                    0xe5a00c0ba65d017d,
                    0x68699ef1ab602535,
                    0x86fb9d7ae465a669,
                    0xb46d9b3cfde2b36a,
                    0xe2ac59715f5d3358,
                    0x02623cbd319e1a4a,
                ])),
                Felt::new(BigInteger384([
                    0x9b9ace11ba45b33e,
                    0xa0cb295ded85604c,
                    0x85bc755dc5133184,
                    0xbe48f964f4bdb3c7,
                    0x075e1764ccfff3fe,
                    0x14a6bca7804dba96,
                ])),
                Felt::new(BigInteger384([
                    0x5f6016c20102a14e,
                    0x82efe9effeb8adfa,
                    0x4723569d10b9008d,
                    0x65fedaf9499f9f76,
                    0xc79b54352b0a08e2,
                    0x03ad9a3c0d1fa3d0,
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
                Felt::new(BigInteger384([
                    0xae10f4334917e471,
                    0x55d9e2e655345b75,
                    0x58ebf6daf9c6e5d0,
                    0x41e5d0fd69b1fe85,
                    0xf51fb003a128800a,
                    0x04a5a0510ddf1248,
                ])),
                Felt::new(BigInteger384([
                    0x7104eb4fd9c467ee,
                    0x18ac4013da17ef2c,
                    0x4354c429165f4f17,
                    0x441c9743b2e80525,
                    0xbcbffab2733e2a08,
                    0x07648d37fd487491,
                ])),
                Felt::new(BigInteger384([
                    0x2ce88c9e2423098f,
                    0x91be57fe2acdfd52,
                    0xefb0e9a9e4e115d8,
                    0x92c8f6cb2e445ef9,
                    0x92c14b7b5e5517b3,
                    0x075d7ebe1fdf645e,
                ])),
                Felt::new(BigInteger384([
                    0x5626e1cd75f9dede,
                    0x43eb57e0dda9b125,
                    0x1443797fb9496c5d,
                    0xd6de92333150f3b5,
                    0x816588d869718fee,
                    0x0aeb67fc6a84caac,
                ])),
                Felt::new(BigInteger384([
                    0x7b062bfc480d38b3,
                    0x05f9ff722eb2a90d,
                    0x6956d859be18b26f,
                    0xa778820166b3374f,
                    0x25f97e4409bb21fe,
                    0x15977edb478903d9,
                ])),
                Felt::new(BigInteger384([
                    0x7cde11ccad925a1d,
                    0x030b3aaa5f70a80c,
                    0xee6d9793abbe4818,
                    0xd065d8f69cc948ea,
                    0xe414c80aa93a6ccd,
                    0x0632b362af1810c3,
                ])),
                Felt::new(BigInteger384([
                    0xa52df40d67d7a352,
                    0xaa6835e7df726e1a,
                    0xda50e72de39974d5,
                    0x93fc8d8906844fdb,
                    0xac9496bf24f2ef3c,
                    0x14e32ffeeef3acf5,
                ])),
                Felt::new(BigInteger384([
                    0x45b6a20dc98dbd39,
                    0xfcf128a4cafe3661,
                    0x914965255042d4fd,
                    0xccdbe51f89ecd3a7,
                    0xc165e3f217ea620b,
                    0x11a2cc72cf834acd,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x96ec7a68e45bd235,
                    0xcb738b5d73fa88b8,
                    0xa412b3b2dcfd37d5,
                    0x0b1b1849928bc731,
                    0x5f1848f85bc8699c,
                    0x109eed7bab5af6bd,
                ])),
                Felt::new(BigInteger384([
                    0xf72c9c87db89d40c,
                    0xd99768c4bf1704fc,
                    0x18bd07b53fd904be,
                    0x15120deead6308c8,
                    0xd3fdc1e161eb1b2a,
                    0x0bfb811d75f04669,
                ])),
                Felt::new(BigInteger384([
                    0xf609b2d5e982925d,
                    0xda401d47db8822c7,
                    0x362897b8817ab465,
                    0x8e135822a4996170,
                    0xb8c03a0a7edd1f35,
                    0x03a510b24c0cde6c,
                ])),
                Felt::new(BigInteger384([
                    0x0bde42fe31db79a6,
                    0x213739293848323f,
                    0x4ea1ce19e9006cb6,
                    0xa8584083f77406a5,
                    0xa62588b0de3f3ca4,
                    0x072883af079890ca,
                ])),
                Felt::new(BigInteger384([
                    0xc95f780eccf32c4d,
                    0x9b03cdd9262c2ee5,
                    0x30ac0ad1a496032b,
                    0xae3da166bc5aad1c,
                    0x0ca8f8ee6b6f90e7,
                    0x0941a505c72ee5c4,
                ])),
                Felt::new(BigInteger384([
                    0xe4e405c3e99bae07,
                    0xd5bb57e392d74a6e,
                    0x8c1781ebdeba23ca,
                    0xd9cea1a332860a31,
                    0x0144eabebecec204,
                    0x117d9d735106d6dd,
                ])),
                Felt::new(BigInteger384([
                    0xe8b44afb2bcc6021,
                    0x75c2938927864ebb,
                    0x209b6c6d0d2937d6,
                    0xd2fd608e40da0ae0,
                    0xa01a329ba6069788,
                    0x0db535b295ceddd9,
                ])),
                Felt::new(BigInteger384([
                    0x01aa89bce931ba7b,
                    0x7c4b06bc9cdd8056,
                    0x88aa958488a72d75,
                    0x41ed105e2d5b3622,
                    0x0126a43a1c852381,
                    0x0f092631d1245e95,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x4115c60cf0c2e638,
                    0x6fc9d5b267e0e459,
                    0x8b9cb97fbd2edff1,
                    0x749df76ec382a22e,
                    0x213bf2f76332eee4,
                    0x0ed192839bd6c546,
                ])),
                Felt::new(BigInteger384([
                    0x86820ec3765bdf75,
                    0xa7446fd6db5e7967,
                    0x0848a91939dae766,
                    0x95091fcbc8b8564a,
                    0x6b8b33c7b9f37c11,
                    0x155c0ae084d2ef24,
                ])),
                Felt::new(BigInteger384([
                    0x321e289163fb3146,
                    0x7a6678970da77529,
                    0x43bbfaf67985a6f2,
                    0x5713bb954d3d1bcb,
                    0x3831eb54740f6253,
                    0x0c8453dcad0b873d,
                ])),
                Felt::new(BigInteger384([
                    0x6f37c9cbcb73c577,
                    0xce57120159ddaf67,
                    0xc3769d1a86917454,
                    0x93eabbadacd80213,
                    0xc1a7f7b0813c83e0,
                    0x083dc06a9d210909,
                ])),
                Felt::new(BigInteger384([
                    0x6777d3587c3529cf,
                    0xce45e82ce705cb1a,
                    0xb381000e87c9f4fe,
                    0x72784ba1aafd4459,
                    0xa9f895855e5f2987,
                    0x00a27cbca6d38216,
                ])),
                Felt::new(BigInteger384([
                    0x8e499e20bf778ede,
                    0xceba7aa3628b7168,
                    0xea2258a2d9a90b74,
                    0xe9ceebd3cf4e0683,
                    0xa0a209d6dae0f792,
                    0x1948ec8a3f2b741f,
                ])),
                Felt::new(BigInteger384([
                    0x5ec489385163876f,
                    0x47c1bd9a52d91796,
                    0x538a137e36ad5f61,
                    0x610dfd98506f5910,
                    0xa0e01df0a25c02bf,
                    0x0483fea46b356814,
                ])),
                Felt::new(BigInteger384([
                    0x0e4ea6645d87dc21,
                    0x53a7bb34ef3f53f9,
                    0xbc282aed1ca7cfc6,
                    0x9d0bbb1317ffa7c2,
                    0x07def471af55cbf2,
                    0x06d9789291aceb1e,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xec82dd819c251e88,
                    0x61adf139fb5f4a38,
                    0xfb0504975762381c,
                    0x310cccb3f102623d,
                    0x7dad5358ab959f2e,
                    0x1262e741d278a989,
                ])),
                Felt::new(BigInteger384([
                    0x4913fb37d2fbb76d,
                    0xbcc59f475ddecc87,
                    0x69b201a5ea96f7fa,
                    0x0c67fd14deaf7378,
                    0x7aafffd6ec69bcd7,
                    0x09b832dd09573bb8,
                ])),
                Felt::new(BigInteger384([
                    0x32f9c405bf7f0ae0,
                    0xd8608d53f9561ef0,
                    0xb7d2f871bc693f13,
                    0x3d33c9f245dccae8,
                    0x4fa14622b33f87f6,
                    0x09434db40d172f5c,
                ])),
                Felt::new(BigInteger384([
                    0xc53bf92c64270d2d,
                    0x066ac1753425f3bf,
                    0x9e54112aea12b5fd,
                    0xa442d3d9eeb6e23d,
                    0x64ce7cf427c5cfb5,
                    0x15743d965bf48aa9,
                ])),
                Felt::new(BigInteger384([
                    0xdb01436065b2b0a9,
                    0x8b238f90ac709918,
                    0xceca71e991248137,
                    0x94d2916b86e707a8,
                    0x11baeac86df0831f,
                    0x131459314c4ba33f,
                ])),
                Felt::new(BigInteger384([
                    0x9d5fb0fd88bb7b97,
                    0xc267aca82bddbf8c,
                    0x8768ca75275a9381,
                    0xdaf0892ede0e94c7,
                    0x5ef3bebec8eb7d35,
                    0x11a397f6c2f3d4bd,
                ])),
                Felt::new(BigInteger384([
                    0x981c25af5b39c189,
                    0x3bef8268805ad87e,
                    0x6ed3929bb1299729,
                    0xa81f065d288f0b9d,
                    0xcdb19df9ee000ed5,
                    0x0ed533ce83e6f7e1,
                ])),
                Felt::new(BigInteger384([
                    0xc93970d322a093e0,
                    0xe48f5d060420ff3d,
                    0x7ae8f889cbc5bd4e,
                    0x00f29d4c2b1324d3,
                    0x7c90fa2a27bd1868,
                    0x1424b38f7e6b6c0e,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x4552f1dd9ae39da3,
                    0x11969fe709cdc404,
                    0xfcded9bfadb14551,
                    0x5b6e70dfb0ebcc11,
                    0x541711e32d7d4a4b,
                    0x1326f2d2a1379057,
                ])),
                Felt::new(BigInteger384([
                    0x019d12f14bd0547a,
                    0xb14c7abe20a3774f,
                    0x5d48277eef8b558e,
                    0xbaa83c470803e031,
                    0x927bdc54862cd5ea,
                    0x150b7cf1b0069e39,
                ])),
                Felt::new(BigInteger384([
                    0x03f8428e587e3312,
                    0x803ae6fe80c7ce42,
                    0x200374c85129d7fd,
                    0xe3b1bad08e1542fa,
                    0xc7c0d08bc2daf09c,
                    0x12cba3105c33bfa3,
                ])),
                Felt::new(BigInteger384([
                    0x2840541a728d3fb6,
                    0x43bc67cd6f8e3ecd,
                    0x032587977aacab30,
                    0xfc0b76c2e33ba0ac,
                    0x31bf24f8960c103d,
                    0x0c68ba9f499e4676,
                ])),
                Felt::new(BigInteger384([
                    0xa0fcf6254d41aafd,
                    0x7bd35364cbba7638,
                    0xde86f164288e8ca5,
                    0x75e48f502c5b2e4f,
                    0xdb49a24f814400a8,
                    0x0ac842cd359e76d2,
                ])),
                Felt::new(BigInteger384([
                    0xc2db9f05f1dd4f66,
                    0xda63930d81bc02d2,
                    0x9b312cbc579732fc,
                    0x231e84fa50fb3837,
                    0x1dc49ef82504e549,
                    0x14be175bb5daff0d,
                ])),
                Felt::new(BigInteger384([
                    0xa468956b442aa8e5,
                    0x9a2acbfd73ca1460,
                    0xf3e671d01d57110c,
                    0xdd1486d487a50f3a,
                    0xafe69665131dbc48,
                    0x12d7cec4431b1089,
                ])),
                Felt::new(BigInteger384([
                    0xc33d5e4b299ed810,
                    0x85f71b9ab9546e66,
                    0x272c5d0138a098a5,
                    0x602c8fa147eb0e6a,
                    0xdd9358232274a2ae,
                    0x1085941c10562086,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x764919b10073b62a,
                    0x3c6bce807ada005a,
                    0x864ce3adff401fda,
                    0x02ff9b5272cdbc43,
                    0xb83e966f0216acaa,
                    0x0af34dfac09d9c35,
                ])),
                Felt::new(BigInteger384([
                    0x761ebf94ef9393ef,
                    0xc19ade5476f15a67,
                    0x42f956f5ea563b48,
                    0x8c037d1fa0705514,
                    0xbce06f45785c1630,
                    0x187d6bf2b03e309c,
                ])),
                Felt::new(BigInteger384([
                    0x3cdec0bb77b9d2db,
                    0x39ce6e796392c90e,
                    0x2dadbc775ed3b2a7,
                    0x1a61bb87ab9147e6,
                    0x4af816b1490dd1a1,
                    0x0d8c95eb722482c8,
                ])),
                Felt::new(BigInteger384([
                    0xf5cda093af0184f1,
                    0x4a1c4fd5c39c2310,
                    0x7f74bcfb47a0ca96,
                    0x58ce316188c9e447,
                    0x58adf4109dfdb6b0,
                    0x068686dd9950d9ac,
                ])),
                Felt::new(BigInteger384([
                    0x52a27d886f3ffab2,
                    0x77c3f4d6ca7a86c5,
                    0x1e1dcec23708f4e7,
                    0xe82c9a7dbffae463,
                    0x296f1e59c54cab57,
                    0x133ca56b1406d608,
                ])),
                Felt::new(BigInteger384([
                    0x3e04dd3373272bb5,
                    0x9ec1cd2ee5ceda7d,
                    0xc9701a6fc371acaa,
                    0x309f589b65160901,
                    0x491de8917fc3975b,
                    0x08f31ceb122d32c8,
                ])),
                Felt::new(BigInteger384([
                    0x2d611e22bcfde3e4,
                    0x7a45d3360904d49c,
                    0x8747a68242491102,
                    0xf5144ccf2a9d847c,
                    0xa9d026b17588da61,
                    0x13adaed247b8f885,
                ])),
                Felt::new(BigInteger384([
                    0x2134a8ecdc7546a8,
                    0xcc8f8008e811c876,
                    0xb39013e23a4229f2,
                    0x05a60f7a2ff5f20c,
                    0x498aa9c1caf770c6,
                    0x028096f5657e1e22,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0xbbcc0bc8ce10a920,
                0x2d74a1c604066308,
                0xaea0f24557e34dff,
                0xa0459d5a0ca5e338,
                0xc2fcb6598ea7fc27,
                0x155b6352a974f46d,
            ]))],
            [Felt::new(BigInteger384([
                0x37edf208ea2980a6,
                0xad8b8f20f59dad2d,
                0xf97e3191553291bd,
                0xd2daca2dde743d3d,
                0xca25a532929dce45,
                0x032a63ac0a025f47,
            ]))],
            [Felt::new(BigInteger384([
                0xa98eb26e9c6d91be,
                0x6d0b839869f87458,
                0xef173d138f5855c5,
                0xbe8cc447858cf12e,
                0xaf6c6c50b63e9a34,
                0x0dfeb5d4ac5584e7,
            ]))],
            [Felt::new(BigInteger384([
                0x39e167485f007273,
                0x5da878ead2ad2e02,
                0xc1b956ac9ec06c9d,
                0x03f79451870a3789,
                0xbdf1f30f0f797234,
                0x1312dacb4e5f5dad,
            ]))],
            [Felt::new(BigInteger384([
                0x745cf19d1e5ad48d,
                0x70733d198ca345b7,
                0xd07e47a86ae05c45,
                0xa9b65b5defe5ba8e,
                0x60ddaf39a637b1e7,
                0x11c0c91400b4d743,
            ]))],
            [Felt::new(BigInteger384([
                0xf7b1c3d4d058f8af,
                0x3c81859f531573bd,
                0x7784b06d0f5dfda9,
                0x78bb85cd907fd032,
                0xfa93d78605803df0,
                0x14b57a7fd0c639e8,
            ]))],
            [Felt::new(BigInteger384([
                0x7b54bc365041d2ba,
                0xf42f801c934645d2,
                0x086acf82e1ecd232,
                0x716edaebe00c105b,
                0xcf1617a4987962da,
                0x16c99ca337c1461a,
            ]))],
            [Felt::new(BigInteger384([
                0xcba4354920a069a9,
                0x57f5e7ff92a54494,
                0x2fedda1beeabf84c,
                0xb8fb8d9e0d2ced46,
                0xe7a121fdde27b478,
                0x11b4136c83bc8bae,
            ]))],
            [Felt::new(BigInteger384([
                0x5598a2aeeeab5958,
                0x75888f68b28d319e,
                0x02785b64a9c2048d,
                0x184832857185cba0,
                0xbd0c732833ef92d1,
                0x12a7eeae82b9dd0b,
            ]))],
            [Felt::new(BigInteger384([
                0x1b30b528675fa42f,
                0x4af2cce849979392,
                0xd4b3d63590635afa,
                0xbfaa84d8f2415d58,
                0x3311e8dc3ad6aa7f,
                0x0c9a3186b1037fbd,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
