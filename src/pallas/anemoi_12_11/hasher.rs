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
    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 12],
            vec![Felt::one(); 12],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
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
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x5a507f9667264cf4,
                0x0b73916adc34c7ea,
                0x878a9f6fc6cd7fec,
                0x24a8b3a3b0f4c268,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x9dbb0641459aef9a,
                    0x87431895f13609f2,
                    0x93f4dfb265457d78,
                    0x195a558047f044bf,
                ])),
                Felt::new(BigInteger256([
                    0x0d20ebfffab21723,
                    0x7dc1b3ef571580e8,
                    0x3b661ab2028133fd,
                    0x25282c15b245ecaf,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x53f61937604b6562,
                    0x6ddba66c6c34dce7,
                    0xf94cd8caac68eb42,
                    0x197f82d19448132e,
                ])),
                Felt::new(BigInteger256([
                    0x1b3c742dd67b161f,
                    0xf18babc97016f7a0,
                    0xa922b596e14ed128,
                    0x1e27c61553fb15bd,
                ])),
                Felt::new(BigInteger256([
                    0x755a0f9a60bd715b,
                    0x28d71d0d4ddfa25a,
                    0x831bf10486838efb,
                    0x2d22bdd66b7a91a3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc8e1429d15c5c923,
                    0xbe73f1b3611154bb,
                    0x3bed2ff93208e0d2,
                    0x080434ca059b47db,
                ])),
                Felt::new(BigInteger256([
                    0xcc6f0b194b7cce3c,
                    0xe4cd2ba0907e02ea,
                    0x861de86b5232b319,
                    0x28a918846a48632d,
                ])),
                Felt::new(BigInteger256([
                    0xbe8f00f80b6e3535,
                    0xc7464db56ab6e12e,
                    0x31f55b93d9651592,
                    0x2b03c3a5f22004cc,
                ])),
                Felt::new(BigInteger256([
                    0x7eec407e20bf1dac,
                    0x0122e75f52524e2a,
                    0xc7ce2253e3a9cd7f,
                    0x1b84670f6b6c9de1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfdb01f4bd2c66c74,
                    0x643808b5b793397b,
                    0x74d3b4caf48036de,
                    0x2616926064f59ba2,
                ])),
                Felt::new(BigInteger256([
                    0x656ae8a942bc392c,
                    0x9393ffad6f72b994,
                    0x382030f16c0c21e6,
                    0x1de128027725d537,
                ])),
                Felt::new(BigInteger256([
                    0xc3b66395acfa16c2,
                    0xefdf854a4e1ffcba,
                    0x8a364dea8bc505df,
                    0x03f2838f53421f93,
                ])),
                Felt::new(BigInteger256([
                    0x01032c2c5bf56c03,
                    0x1ba1a0fb70662d00,
                    0x452f981422b90091,
                    0x0bf8e90f3b0fe492,
                ])),
                Felt::new(BigInteger256([
                    0x90e0fd86ead750f5,
                    0xe4a871d7b133a6d1,
                    0x6d5bcaaff6e1779a,
                    0x3f1480ce27acd531,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x12f5af7a35a153b0,
                    0xb01cfb6093cc04c8,
                    0xd905974017351c9b,
                    0x389bc31dac10783d,
                ])),
                Felt::new(BigInteger256([
                    0xf6c6fcb8b4400071,
                    0xf18d6deeac6dee54,
                    0x06ecbf9a43e6322c,
                    0x2fffaa0f22215f0f,
                ])),
                Felt::new(BigInteger256([
                    0x71f39685b4c6b9de,
                    0x3e39eb6b924eb66f,
                    0xd07db14122bc8d23,
                    0x0c9ccac247a378e8,
                ])),
                Felt::new(BigInteger256([
                    0xb939dde93ea466d9,
                    0x756595a53c5117e1,
                    0x5ef6d477d17921d1,
                    0x32636abc7e2eef10,
                ])),
                Felt::new(BigInteger256([
                    0x6fbd7f3381151c7a,
                    0x929c7c4c01c54345,
                    0xac2fbe81762c39f3,
                    0x3e393f99bf836696,
                ])),
                Felt::new(BigInteger256([
                    0xf749121921215fe3,
                    0x87aaa3732262f49c,
                    0xa9cc73160dfcb66e,
                    0x3bfdc3dc0a4612e5,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x5242e4fd780d5563,
                0xf72ca3ffe1669834,
                0x5649c9d82398086d,
                0x2db6556d6dd018fc,
            ]))],
            [Felt::new(BigInteger256([
                0xbaacab65761daadf,
                0x36c44a2821327b3b,
                0x5428f5ddd6ddfcef,
                0x19e3afde1a771787,
            ]))],
            [Felt::new(BigInteger256([
                0xc7185ce78d2b9a72,
                0xc90f4ec706d3f962,
                0xe009d36b352c726e,
                0x278fb17e555dcfa2,
            ]))],
            [Felt::new(BigInteger256([
                0xb504b935ca021e2a,
                0x5675e2fb7f1e03a5,
                0x1f160a8332c7fee5,
                0x0303500da3e12265,
            ]))],
            [Felt::new(BigInteger256([
                0x2d996d2cb39e143f,
                0x1aa60cb7c297371f,
                0x7c3116b404ea188c,
                0x04d71e40866e2b32,
            ]))],
            [Felt::new(BigInteger256([
                0xd799cce46d7f3fff,
                0xeb1264d65249936f,
                0xf93243adf33880f8,
                0x215eaf19479657df,
            ]))],
            [Felt::new(BigInteger256([
                0xceef5e452d993ab3,
                0x59dff977ccce2083,
                0x1d35aa2ed65bc6cd,
                0x35ccd712cadaeabc,
            ]))],
            [Felt::new(BigInteger256([
                0x8a3040e5271a4039,
                0x0e27dc9e98aef7cb,
                0xcbc964e4f1a692c5,
                0x293e2ac2b5a33a3d,
            ]))],
            [Felt::new(BigInteger256([
                0xaf3cc2045e85cd5f,
                0xed1a9ca16936d01a,
                0xb104fabb6e251c4a,
                0x0c15c74e01fd1dfb,
            ]))],
            [Felt::new(BigInteger256([
                0xa8bcd0a3c0167221,
                0x05278d4b4379285d,
                0x559b5b5e2cd7c2c8,
                0x08d5b813f7a03c61,
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
            vec![Felt::zero(); 12],
            vec![Felt::one(); 12],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
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
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2262e460852a30a3,
                    0x1f5fd61d906b32e8,
                    0x562b225ce9d50365,
                    0x2404049a11bdc992,
                ])),
                Felt::new(BigInteger256([
                    0xed9611f18315b4e8,
                    0x9fb0681fd39e0da1,
                    0xf50497edd09538a6,
                    0x3150d71ee3a324f0,
                ])),
                Felt::new(BigInteger256([
                    0xedff729b8fdb634c,
                    0x800f345020d7125a,
                    0xea38906aacea97cd,
                    0x177188e488410f83,
                ])),
                Felt::new(BigInteger256([
                    0xdb3abb482d4f1f9e,
                    0x36194f49344a5a33,
                    0x97e5c9db82cf167c,
                    0x28ed70ef9dcf7b13,
                ])),
                Felt::new(BigInteger256([
                    0x7d8e8a81b0dba793,
                    0xf8450e66410b57b6,
                    0x4ea526d91b9f5ca3,
                    0x1f7ad353020d96f7,
                ])),
                Felt::new(BigInteger256([
                    0x255c86e583bcc08a,
                    0x82c16ef6b9f75297,
                    0x5d00253c48a6134d,
                    0x3ecc7a9535f9205e,
                ])),
                Felt::new(BigInteger256([
                    0x5977165d69a1e188,
                    0x05e977c577461e03,
                    0x951e7b71e651eea7,
                    0x39824be7e0902cf4,
                ])),
                Felt::new(BigInteger256([
                    0xf1fc37f707485c38,
                    0xc86e602fbe79b235,
                    0xc2c75e32e38c85bd,
                    0x1e4dd374e4d4fd0e,
                ])),
                Felt::new(BigInteger256([
                    0x848583054a17f5e5,
                    0xeef46fc3a2e9aaf2,
                    0xa238cf6ada4258ed,
                    0x05607a46fb9272b6,
                ])),
                Felt::new(BigInteger256([
                    0x27eaddf81e780cf5,
                    0x246b47b9af692024,
                    0xca68f22923ca7efd,
                    0x3770ce2032ebb416,
                ])),
                Felt::new(BigInteger256([
                    0x71274ed4c2fd0066,
                    0xc59b5bfcc6be186d,
                    0xf75465afd80e98a8,
                    0x19e65d601354b254,
                ])),
                Felt::new(BigInteger256([
                    0xa6fe26d11573c883,
                    0x7b951b1403500282,
                    0x6b2429ff75daeeba,
                    0x1f2e1000e03af730,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x586ba61b5cb22505,
                    0x956bc18125696466,
                    0x1df05ab728ebe01c,
                    0x340f38f5e6c5da41,
                ])),
                Felt::new(BigInteger256([
                    0x2114675155cd2c2a,
                    0x883389e930aba0ac,
                    0x7c9a97763a7821ea,
                    0x1ecba96c51015da8,
                ])),
                Felt::new(BigInteger256([
                    0xa4a97c1026116577,
                    0x6ff7ab44119ebf06,
                    0xcba453ad60e22633,
                    0x0637d45e63c7d0c6,
                ])),
                Felt::new(BigInteger256([
                    0xd7590a74016c75e6,
                    0x79db7a1e583b8809,
                    0x604b1713620b6e4f,
                    0x068dac4d51decbb8,
                ])),
                Felt::new(BigInteger256([
                    0xbc16ebbb050332ab,
                    0xd89e070b26f24512,
                    0xbacf849cb2d80abb,
                    0x1a37dab045ff2381,
                ])),
                Felt::new(BigInteger256([
                    0xb19eb545642d05f1,
                    0xc12b49e9a1006088,
                    0xbca83ff5cb94cea7,
                    0x0b2c4feae77826c3,
                ])),
                Felt::new(BigInteger256([
                    0xa7794cdf86d468c5,
                    0x92d002ff57beaf2d,
                    0x2d2e3a50810607d6,
                    0x06d032782cdcd5fc,
                ])),
                Felt::new(BigInteger256([
                    0x7c7c19aecde26418,
                    0x0666703359a28784,
                    0x15f1552d76ec9f4e,
                    0x0d0128a7ec86be5a,
                ])),
                Felt::new(BigInteger256([
                    0x4af7ecfa40459966,
                    0x31155daa1020e531,
                    0xd7779fdfab7fdc76,
                    0x1dfe762a88e6da74,
                ])),
                Felt::new(BigInteger256([
                    0x2fb0417603684b67,
                    0x2d092a33d8ec706c,
                    0xfb3e43ba60c3264c,
                    0x19dbbbe539d58320,
                ])),
                Felt::new(BigInteger256([
                    0xc1388bed4a9eaa6b,
                    0xb77033a243b71ce6,
                    0x627ff5e9de7ed71e,
                    0x02e5f12a1693098a,
                ])),
                Felt::new(BigInteger256([
                    0x9fb4fe68bd7f549c,
                    0x126995cb1498a7dc,
                    0x2eedb1421e88545c,
                    0x16b50e3922bc9c20,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf204cadf27538258,
                    0x2b0130bab11fea02,
                    0xa24b7152a8d7f9b9,
                    0x228cdcd351331214,
                ])),
                Felt::new(BigInteger256([
                    0x73ed060b6befa2ef,
                    0x3c94717b54852512,
                    0x0ce5c1a33f100e75,
                    0x0f3ffd177c69f93e,
                ])),
                Felt::new(BigInteger256([
                    0x89a584ac927038ad,
                    0x85e0ba17039c9146,
                    0x819715c0c011c3f2,
                    0x05d5551f1c7777c0,
                ])),
                Felt::new(BigInteger256([
                    0x4548b2946d900ef6,
                    0xf3c9fb08538c149a,
                    0xb3dbe96c4e87f741,
                    0x0efec2d2ea3fa0bb,
                ])),
                Felt::new(BigInteger256([
                    0x64b54d11a40d1648,
                    0x371db7713450dfa7,
                    0x2e23615e39593489,
                    0x3267b90158625299,
                ])),
                Felt::new(BigInteger256([
                    0x8f786a9239818a99,
                    0x030a7d3068c339df,
                    0x595314408b7184a0,
                    0x35a6a3d3d03371fa,
                ])),
                Felt::new(BigInteger256([
                    0x85b57db42e0a3cea,
                    0xe6af80f57f9694a2,
                    0xf254602fe0542bc0,
                    0x3d0a8441cdea1eaf,
                ])),
                Felt::new(BigInteger256([
                    0xdbb6624a02c9fed5,
                    0x5382c0a3ceea9050,
                    0x7c14cb921346f864,
                    0x0379709c4b0ace9d,
                ])),
                Felt::new(BigInteger256([
                    0xdb7a688ad4834efa,
                    0x0f796c736dce250b,
                    0x21213da4e097fad8,
                    0x2eeb5203b7987cf3,
                ])),
                Felt::new(BigInteger256([
                    0xfe922b085cf11adb,
                    0x4b92fae8ff952eab,
                    0x31f48e3dc902f2fe,
                    0x06e943fd30e0258b,
                ])),
                Felt::new(BigInteger256([
                    0xce096025d60e6ef3,
                    0x9fde06f7a9a43cec,
                    0xbc88b949b1863158,
                    0x386a79eb1614f564,
                ])),
                Felt::new(BigInteger256([
                    0x8684df05b0d05cf9,
                    0x8d4577559a1fddd6,
                    0xeb8489e44107ea98,
                    0x15eb38eb325d8653,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbdcebda5262e0601,
                    0x9b76d9b55377f5c9,
                    0x752065ddfaafe5a8,
                    0x01a3e236da5ec108,
                ])),
                Felt::new(BigInteger256([
                    0xd355b06b86e729dd,
                    0x7bec6e3cade84c68,
                    0x10f9bbb767f7ab88,
                    0x2d3c2607d651ea85,
                ])),
                Felt::new(BigInteger256([
                    0xda5a37fdd3334d54,
                    0x08bac1c503680cf9,
                    0x2e7c28671fb98cbe,
                    0x1a31c8177feb413c,
                ])),
                Felt::new(BigInteger256([
                    0x99aed540a31086b0,
                    0x24d9a3aaecc2878c,
                    0xc0752492e68b8994,
                    0x12bd0d27a5141de5,
                ])),
                Felt::new(BigInteger256([
                    0x9fc5e57c7af402f9,
                    0xd355a5b65b02f1b9,
                    0x847e14579489d91d,
                    0x289fb4d99570fde6,
                ])),
                Felt::new(BigInteger256([
                    0x379921204ef6ad90,
                    0x838efd1581fc92a7,
                    0xdca8c1cad65660af,
                    0x1c23587f69868c64,
                ])),
                Felt::new(BigInteger256([
                    0x665d7c8571d5f35e,
                    0x3164ccb08e1c8130,
                    0x7399a1433ce428ce,
                    0x1db0bba9947a0e95,
                ])),
                Felt::new(BigInteger256([
                    0x59cc3fcc491bdb48,
                    0xde4e9afe405234f3,
                    0xaee98b7df0a9cdcb,
                    0x2fb5ce8fe0c3dc57,
                ])),
                Felt::new(BigInteger256([
                    0xf2895155b8323284,
                    0xd9e3067d8eaaea17,
                    0xe9fd626595f898d7,
                    0x37704dcb239595a3,
                ])),
                Felt::new(BigInteger256([
                    0xc7297ef11545bd3e,
                    0x6a523942416724b8,
                    0xcbd85653cddbf7c1,
                    0x211a2cd886956b5d,
                ])),
                Felt::new(BigInteger256([
                    0x585d44a5b299f1a8,
                    0xc6fe4d22b9e0218a,
                    0x6815bf5e923639c3,
                    0x0d07c72a1f1bcd35,
                ])),
                Felt::new(BigInteger256([
                    0x636fb257ca46cb85,
                    0xa799e2bf6fc7cbcd,
                    0x7d7b12f5420cdcfc,
                    0x13980d2e4af8a3a1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc37bf109fd8e4cfe,
                    0xfe7cd89ee3064ef1,
                    0x3cc9bd4b3caa4e2d,
                    0x2f5e2d267d5149fe,
                ])),
                Felt::new(BigInteger256([
                    0xfd3d812f16ab7747,
                    0xc241ecfa12896fa6,
                    0x9e3e76491798d431,
                    0x11f0314d73eab0b2,
                ])),
                Felt::new(BigInteger256([
                    0x5e41d09a3c90694f,
                    0x192b2bfff04c77c4,
                    0x20e0be317cf1f3ee,
                    0x265d5e389135c57e,
                ])),
                Felt::new(BigInteger256([
                    0x5d82529a98d22f93,
                    0xe66d51b8feccf9c1,
                    0xb89ec882013823cd,
                    0x004b5fb5e96d9383,
                ])),
                Felt::new(BigInteger256([
                    0xfd6012588ec92789,
                    0xdb97277d040e7ca1,
                    0x7e90ebb5fd3a38c7,
                    0x3551fbb484d3c923,
                ])),
                Felt::new(BigInteger256([
                    0xcf0eed43e22df5a9,
                    0x248ef30102cc718c,
                    0xdf381b6b2ad354a3,
                    0x27811a25c3d2c698,
                ])),
                Felt::new(BigInteger256([
                    0xac3d89b920cb8e47,
                    0xa03579d111cde1ac,
                    0x76b4ca882e206d34,
                    0x0b17e14718bed73b,
                ])),
                Felt::new(BigInteger256([
                    0x22ed3e730fc9d6eb,
                    0x16181749f7e024c1,
                    0x01396cf9cc093e88,
                    0x3a9610b3929a65fd,
                ])),
                Felt::new(BigInteger256([
                    0xea63c316382224d3,
                    0x6d3f28d94885ff19,
                    0x83606ac4f62c9a68,
                    0x34ce349098e212f2,
                ])),
                Felt::new(BigInteger256([
                    0x4995f2599a386fea,
                    0x288f4b081b654795,
                    0xdf69c1427b5e6889,
                    0x30661c9b26023a61,
                ])),
                Felt::new(BigInteger256([
                    0xa54bb9f92a17b64e,
                    0xff9e2382897d2f03,
                    0x1dd4c13406642a74,
                    0x0cdc57f5e0d8caa8,
                ])),
                Felt::new(BigInteger256([
                    0x67cbbd37fb0e2caf,
                    0xe40840f4735c436b,
                    0x718a828a264f009d,
                    0x0c00c2d95e898adc,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x17d5f1466cd22f9e,
                    0x17a7c118a18da0ff,
                    0xb035880e188cad4a,
                    0x3247e32e229fe334,
                ])),
                Felt::new(BigInteger256([
                    0xffe312223848815c,
                    0x758da6bccb694d95,
                    0x94cd90edabe53a1d,
                    0x14b3b90894cd7306,
                ])),
                Felt::new(BigInteger256([
                    0xc2543568992014d1,
                    0xe66fb56db677195e,
                    0x9ec49f064fe0824b,
                    0x151579fdb507b528,
                ])),
                Felt::new(BigInteger256([
                    0x63a0d3469e209f87,
                    0x1abb54a4ec5db170,
                    0x7f03fa7d95bcf761,
                    0x1cfb73fb8f32bfd3,
                ])),
                Felt::new(BigInteger256([
                    0xb5c3517fa0a0dbab,
                    0xdba8731a08acbe4d,
                    0xcabea1e0b9f15d6a,
                    0x385a03153574fdc6,
                ])),
                Felt::new(BigInteger256([
                    0x46bc17227fa12c0f,
                    0xb3f32682b0d7bfac,
                    0x08015b08e6251dda,
                    0x213c79dc8ac68a58,
                ])),
                Felt::new(BigInteger256([
                    0xd746d460042e532c,
                    0xaf5afb0b05a63511,
                    0xf244e67e45f511ea,
                    0x0eada188706f9c23,
                ])),
                Felt::new(BigInteger256([
                    0x835927a2cdea1aea,
                    0x73fa769f967fc070,
                    0x89ff993a71a5ea14,
                    0x3ad1bddaa6f9c0fa,
                ])),
                Felt::new(BigInteger256([
                    0x00a1acdf0c92a751,
                    0x7f84523549db59dc,
                    0x1ca69c04f5eac510,
                    0x3d1c92d7f02ba685,
                ])),
                Felt::new(BigInteger256([
                    0x51ab174b39b6e119,
                    0xfaa00362a4bd7fc7,
                    0x7d27e8d27d6d9d35,
                    0x27211700744ad258,
                ])),
                Felt::new(BigInteger256([
                    0xcae1b984c83cfa14,
                    0x7d166b2eccd2fd8f,
                    0x37f163f6adae7d00,
                    0x151944a535acdf3a,
                ])),
                Felt::new(BigInteger256([
                    0x015b288b577e68c1,
                    0x871261dcaf5c9700,
                    0xeb481b9d563add1c,
                    0x12b03fa9d1b7cd7f,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x14da49e76fb744d1,
                    0xdf27979d767cfff7,
                    0xa7c1d92ef4b3b98a,
                    0x3040219edcee9a90,
                ])),
                Felt::new(BigInteger256([
                    0x09157eacc5bc651b,
                    0xbb154a4940bd3092,
                    0x124137cd4a679b20,
                    0x360a23432ba782c0,
                ])),
                Felt::new(BigInteger256([
                    0x8edffcf56293c00a,
                    0x83b4bddf016f327d,
                    0x4380a37f537cecb4,
                    0x2158315ee62c773a,
                ])),
                Felt::new(BigInteger256([
                    0x6fa423f74f726dd9,
                    0x0297b9288e5909e4,
                    0x1939078737fde125,
                    0x234520fba2ef8184,
                ])),
                Felt::new(BigInteger256([
                    0x1b85915a4ab5fa48,
                    0x5ca8c7e8170340d8,
                    0x8b83f4436f52efe3,
                    0x2554f92300302760,
                ])),
                Felt::new(BigInteger256([
                    0x0edac0d2f105e398,
                    0x83346e94d51cacbf,
                    0x17036d147ab8fc7e,
                    0x3685d6166d5ce4cf,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9e99b4be10d302f2,
                    0x37902b173a6d7ad7,
                    0x46016684ff07e342,
                    0x3dea62e6d28ffd1f,
                ])),
                Felt::new(BigInteger256([
                    0x033ec80e50ef18d5,
                    0x7f56ac7301c12709,
                    0x92ec09838f6fee2b,
                    0x1d880e6b8ca12c48,
                ])),
                Felt::new(BigInteger256([
                    0x8f42c64a34523253,
                    0x58b29035a57f6c76,
                    0x2d558e280514d816,
                    0x2a2260ca61f93c97,
                ])),
                Felt::new(BigInteger256([
                    0x9719d5a4c243e166,
                    0xdd520874809f9f86,
                    0xc74160e61dc285ff,
                    0x2700bea023fe2879,
                ])),
                Felt::new(BigInteger256([
                    0xd53b2863e5756ce1,
                    0x220cab18fff97cbe,
                    0x331cef37d7d0b374,
                    0x115479556a8d7f55,
                ])),
                Felt::new(BigInteger256([
                    0x91f8183d27b2a7bd,
                    0xeb63a50b4caae63a,
                    0xc1af7bd6878ce042,
                    0x3009b06d29b34e0d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x530cb9a406f10f78,
                    0x07d2f13c209b11a5,
                    0x14b96145e6663285,
                    0x0af3f068b95618a4,
                ])),
                Felt::new(BigInteger256([
                    0x0ce175f2d2b70d10,
                    0xa0c7ccfa8ddc3009,
                    0xd0b070af04a7b4d2,
                    0x01623f54da9240dc,
                ])),
                Felt::new(BigInteger256([
                    0x6d4a12061652687b,
                    0x2147996867273a45,
                    0x2d22f47f3e4be711,
                    0x22ba221256cab907,
                ])),
                Felt::new(BigInteger256([
                    0xb4eb585d227cb107,
                    0x4e2f9cde4eb03719,
                    0x7cb81b7b51b28626,
                    0x078874fbc8333191,
                ])),
                Felt::new(BigInteger256([
                    0x5b1c13da6f0d962b,
                    0xbb7bb9ca864def17,
                    0x659803f6e3c41141,
                    0x1f9dbe16e24d8458,
                ])),
                Felt::new(BigInteger256([
                    0x41e5b5784ef556af,
                    0xcbd632756d46cc15,
                    0x9a24bd8c2aa74d79,
                    0x244eba27778af4d2,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe0d58412287d3017,
                    0x0e7d242fe2da6a38,
                    0x183b3427d9c639f9,
                    0x2be9b41d89aa1d1c,
                ])),
                Felt::new(BigInteger256([
                    0xff54e95bf257a191,
                    0xed34cda9c98c29c8,
                    0x72efd932fb6c6b7f,
                    0x0d938c746de2151a,
                ])),
                Felt::new(BigInteger256([
                    0xcf18aaeb64f7dfbb,
                    0x4e0ba170867a86e0,
                    0x964e254ff57b7f9e,
                    0x1c46899242569c11,
                ])),
                Felt::new(BigInteger256([
                    0x10721f48733a9c75,
                    0xff48c3fb626ddac9,
                    0xe8f68d681ce9c1f3,
                    0x080225fbed4dba85,
                ])),
                Felt::new(BigInteger256([
                    0xf74959e8b0223768,
                    0x0a7f413a795627b6,
                    0x9ff22fa195238a1a,
                    0x333ec26d6bd3fe19,
                ])),
                Felt::new(BigInteger256([
                    0x5f1b8841f3bc2ddf,
                    0xb5c685a740e39939,
                    0x07b4388ba58cd805,
                    0x0bbc4cb9c71f45b6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3577af33fe6a658e,
                    0x42787188033e04e8,
                    0x4e740099bf8c3d9a,
                    0x10e6810cfb291c68,
                ])),
                Felt::new(BigInteger256([
                    0xcadc33d5539dcb82,
                    0x062eade0c2abec06,
                    0x767a7195e8cf9b20,
                    0x0b7f74162a1f4801,
                ])),
                Felt::new(BigInteger256([
                    0x5a3f3108ec6eb673,
                    0xe2a19c80740dbeec,
                    0x6132b6ebdeb44ecb,
                    0x3e1286095a682f19,
                ])),
                Felt::new(BigInteger256([
                    0xa5169a1038b14a50,
                    0x187a47412743cc58,
                    0x4870bf43e78133ae,
                    0x02f381224557ec94,
                ])),
                Felt::new(BigInteger256([
                    0xfafba2753973760f,
                    0xfe919ec328760390,
                    0xfbf21013cf4f3a72,
                    0x21c4b32858f979fa,
                ])),
                Felt::new(BigInteger256([
                    0x1257b48f224cfabb,
                    0x721085af525cd996,
                    0x016eabfb5a780e45,
                    0x1cb3caa18be134eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb074d24597745422,
                    0xbbb424e790d6bbb7,
                    0xd1792fada0573878,
                    0x0148bc244b5f384b,
                ])),
                Felt::new(BigInteger256([
                    0x75003ed5d25cac4b,
                    0x776587c55bd6ea1e,
                    0x246dbf75b3cbde83,
                    0x341c39229176e4d0,
                ])),
                Felt::new(BigInteger256([
                    0xb50e67d812ccd6c1,
                    0x182a47844bbdf3ec,
                    0xba5d4b86d92966ef,
                    0x11f2c4aa12cda7b9,
                ])),
                Felt::new(BigInteger256([
                    0x68b907464b72e328,
                    0x63b52c06b7659eaa,
                    0xf11414e24a3047c8,
                    0x360c7556edefa9df,
                ])),
                Felt::new(BigInteger256([
                    0xd9f2ed3a4756e04f,
                    0xcd8a83046bba7d0d,
                    0x8c574a8915ce0ea0,
                    0x02723d923b53b0db,
                ])),
                Felt::new(BigInteger256([
                    0xcb627f2628c61d36,
                    0x74b3bd262bb4ad94,
                    0x9d1bcb668c7043dc,
                    0x24da6669e630322d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x8b478f2be059258d,
                    0x34ce6f7533803d46,
                    0xb98f72ecab80be91,
                    0x077ef2eaded69de5,
                ])),
                Felt::new(BigInteger256([
                    0x150179378402046e,
                    0x71c477651037c0b7,
                    0xe3ed2cfd90d1c44a,
                    0x20d22932de618eb4,
                ])),
                Felt::new(BigInteger256([
                    0xec2419d96f6eb70c,
                    0xeafc70ebb0a6df21,
                    0xfc6d7e0f66b57d75,
                    0x21895788815ad1a6,
                ])),
                Felt::new(BigInteger256([
                    0xe251b4a3c0f9827f,
                    0xabb5b0aff4b25f2b,
                    0x80a3fc4356e739c8,
                    0x346c201eafe4d87a,
                ])),
                Felt::new(BigInteger256([
                    0x7de1ef20ab37d563,
                    0x167934a351e7ff74,
                    0x12adad2b47752cd6,
                    0x1472817ffc752c8b,
                ])),
                Felt::new(BigInteger256([
                    0x5df09a86b5082ff0,
                    0x654cbfba8a57e100,
                    0x00b71168de4f9404,
                    0x34f7765a3cd2407e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf03a9819917b97e3,
                    0x37ad7b046ca95636,
                    0xd6ea7e9e8635abed,
                    0x0106ce208647bde1,
                ])),
                Felt::new(BigInteger256([
                    0xcc57b612e77039ca,
                    0x828aa9427415ad01,
                    0xee218090969cd44d,
                    0x2c9ca48972fc8e74,
                ])),
                Felt::new(BigInteger256([
                    0xe7414ce141aadbcb,
                    0xcb589909eab6ba7a,
                    0x1dea3effd52c2da2,
                    0x36cf2341869b09ae,
                ])),
                Felt::new(BigInteger256([
                    0xc67de6283966af22,
                    0xd8049919a5451283,
                    0x0dee1358d42ca89c,
                    0x11ec7e2b291aa329,
                ])),
                Felt::new(BigInteger256([
                    0xf0baef93e7e6e41d,
                    0x3f7a1d54c9e5b92e,
                    0x2a8202b263cdde5e,
                    0x05e9b82384fe55c6,
                ])),
                Felt::new(BigInteger256([
                    0xbb7f961328858316,
                    0x2d3ca4c8d369504f,
                    0xdb0d7ffa387da2b4,
                    0x2f171195dc9e0c29,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf512c46dac662a26,
                    0xbf867addd6a53670,
                    0xfdb451949642a7f1,
                    0x1788e548ddb6c16a,
                ])),
                Felt::new(BigInteger256([
                    0x05c4126a3c59aa97,
                    0x4b64fde52a0f2f59,
                    0x0ca9dccf13216d62,
                    0x1fc31912fd268fba,
                ])),
                Felt::new(BigInteger256([
                    0x4497773d18f88c33,
                    0xc8cef1bebbe29b74,
                    0x8a1f8422c2488d83,
                    0x3cd07add88494a75,
                ])),
                Felt::new(BigInteger256([
                    0x64755104e0db510b,
                    0x4da876764672378f,
                    0xc8f60a9566616797,
                    0x265773c20ca10f93,
                ])),
                Felt::new(BigInteger256([
                    0x25ad39b183066807,
                    0x0218dd72ff0299ff,
                    0xab162da24fdc7f1d,
                    0x3d9ebbe5aaa4809a,
                ])),
                Felt::new(BigInteger256([
                    0xe397a12fb46c77bf,
                    0x1d446dc3431f248e,
                    0x1748fd99830eb2f8,
                    0x280062b65de3dbe7,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9786fd97cf5d1ae2,
                    0x8d93abdd508900f6,
                    0x34448c1098c6b0ad,
                    0x32d093346eaa0160,
                ])),
                Felt::new(BigInteger256([
                    0x58f883f3dd938992,
                    0x07f0ce9a0e5798e5,
                    0x32240eb6127f197f,
                    0x102d2ab8f68a34dc,
                ])),
                Felt::new(BigInteger256([
                    0x590187806feb9a8f,
                    0xb5c862bc3ae197ec,
                    0x8587e9ff833ac7b5,
                    0x2e9a036f40340f3d,
                ])),
                Felt::new(BigInteger256([
                    0xcac06a6f707d0ccb,
                    0x7bde94478d32773b,
                    0x1e8604167311166a,
                    0x3e6b909355c4f128,
                ])),
                Felt::new(BigInteger256([
                    0x12ad3ab45b1090b7,
                    0xccbf2716598d8435,
                    0x6bc67f2f33490405,
                    0x0609a5f9139eb98c,
                ])),
                Felt::new(BigInteger256([
                    0x3cd426473b402f04,
                    0xc2c7827b83e684ff,
                    0x93ca462d4436cbbe,
                    0x13f975c52cdf7820,
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
            vec![Felt::zero(); 12],
            vec![Felt::one(); 12],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
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
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd331bb8f7e02563c,
                    0x603bcfe72a3c8b98,
                    0x7c73347dc2fed284,
                    0x2d5faa75677ca5c3,
                ])),
                Felt::new(BigInteger256([
                    0x7df15409ff4e0a44,
                    0xc91c78a6af2657f5,
                    0x55b0f3c083379455,
                    0x38a71856513856d8,
                ])),
                Felt::new(BigInteger256([
                    0xa46de9ce4f0be197,
                    0x071f88e825f81c69,
                    0xdc78f3d927e8feb3,
                    0x212fb069b0f9cbda,
                ])),
                Felt::new(BigInteger256([
                    0xac4e34ff32ab58fd,
                    0xd520cd5e2fe07de8,
                    0x14b7a92a2605e849,
                    0x39bd6d970f551d78,
                ])),
                Felt::new(BigInteger256([
                    0x9438ec76fa022c73,
                    0xe1c3f29d4f8e6d36,
                    0x291d89206ab8ec06,
                    0x158a3693803967f2,
                ])),
                Felt::new(BigInteger256([
                    0x93c0831b9257383e,
                    0x0cad3c5979111b38,
                    0x4be709947f46a444,
                    0x207bbb5a138152c4,
                ])),
                Felt::new(BigInteger256([
                    0xfb939ef1de8ff98d,
                    0x50836ca0d796c439,
                    0x3459f6ec1bc6bf21,
                    0x271e1a8b7a80ff05,
                ])),
                Felt::new(BigInteger256([
                    0x1a1ea92f1659969a,
                    0x25ecf5074da202f6,
                    0x9c75b0b53d590c7f,
                    0x0aceb0933eb82847,
                ])),
                Felt::new(BigInteger256([
                    0x9222dcdad35dcc5c,
                    0x458591356b80ec87,
                    0x02396190dad3babe,
                    0x161d7f282a468565,
                ])),
                Felt::new(BigInteger256([
                    0x7bfef34b24bca558,
                    0xe4c5bce6647a7e1d,
                    0x5e476ef71c8e1cff,
                    0x08b10cce9ccaee2c,
                ])),
                Felt::new(BigInteger256([
                    0xba64e2ba0a237628,
                    0x38fc8365e515a086,
                    0x9da769d33d3a45d6,
                    0x21a432b6ef5d74f6,
                ])),
                Felt::new(BigInteger256([
                    0x899dc208cfa26e97,
                    0x8310a8e65589fb56,
                    0x737bd47c702d9035,
                    0x3bf482f2fb0ff0ef,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf5730138db1e8e6c,
                    0xd6f6720a4b3837e4,
                    0x7b8811ed768a4bc9,
                    0x2b93bee8990392a5,
                ])),
                Felt::new(BigInteger256([
                    0x8255873065428fae,
                    0xfbc86197d5393b6f,
                    0x0ad2556dfaca265f,
                    0x2646d452aa91699f,
                ])),
                Felt::new(BigInteger256([
                    0xc291d5f894b8aea3,
                    0xd27d458d7aa4d061,
                    0x04936cfd3f172077,
                    0x2a47254b34b17822,
                ])),
                Felt::new(BigInteger256([
                    0xad9f703ddbd6ce74,
                    0x9c61d2cb179a3674,
                    0xff983ab6fe4549e0,
                    0x0a0a5f657d02c4a1,
                ])),
                Felt::new(BigInteger256([
                    0x22fe3ee131c01317,
                    0xe90ab281333448de,
                    0x9f3e6f95d647559b,
                    0x13c5db42d9ffa1fc,
                ])),
                Felt::new(BigInteger256([
                    0xb322612f59c7cc65,
                    0x18725b2d6c6fe6cf,
                    0x00a83e1993a90342,
                    0x367ff6effa6b4b22,
                ])),
                Felt::new(BigInteger256([
                    0x49d29aab4afe50a0,
                    0xedcc4a109a3d6399,
                    0x7a63b5173c4fc65e,
                    0x323c7b86b60541c4,
                ])),
                Felt::new(BigInteger256([
                    0x0ca48d71c71009b6,
                    0x74e91766379afe4d,
                    0xb5c46c1c6d212276,
                    0x2c2a7f8403505a09,
                ])),
                Felt::new(BigInteger256([
                    0x51eed8795cb870e7,
                    0x1a72f4aa124b5d6c,
                    0x82e27c18d29c1f46,
                    0x2402d116656225c0,
                ])),
                Felt::new(BigInteger256([
                    0x6fe1fd86dbb32060,
                    0x9bd5171e97ea7e84,
                    0xf56fe98789364036,
                    0x32fce0aa42ef3cad,
                ])),
                Felt::new(BigInteger256([
                    0x59c6ada7a6b91888,
                    0x5abdea36dbf339d2,
                    0xfd686db432bb68a3,
                    0x2ce7c40dc19f1219,
                ])),
                Felt::new(BigInteger256([
                    0xb47cdce9708373dc,
                    0xb5f983f2504e28c4,
                    0x2d9e10e6e9d72610,
                    0x01cc2026e5487d37,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x41da77d350e7062e,
                    0xabff79e28459a3ef,
                    0x0ebc1ea97b9c743e,
                    0x1bf198942aeb78e0,
                ])),
                Felt::new(BigInteger256([
                    0x8ebd3df940ba3a30,
                    0x967660b456c482d1,
                    0x8b292452fe357b5c,
                    0x1eb77fbb304a305e,
                ])),
                Felt::new(BigInteger256([
                    0x00e28eba8a868e31,
                    0xdc9346d022c8c392,
                    0x979cc41fc4fcccea,
                    0x118d7f6781991664,
                ])),
                Felt::new(BigInteger256([
                    0x5579a86cafb492e8,
                    0x0281390d5162f95d,
                    0x4a66a26f0bc5e313,
                    0x223a91b3dad130bf,
                ])),
                Felt::new(BigInteger256([
                    0x0cb707c3c21c0107,
                    0x34ea085b9ef8bbbd,
                    0xf748c3fea90a106c,
                    0x02ef2b8fa5d189c2,
                ])),
                Felt::new(BigInteger256([
                    0x4f7d596516c1a33c,
                    0xaee628503d8d746c,
                    0x9798ef187a998d9b,
                    0x03b5b770feeea0ca,
                ])),
                Felt::new(BigInteger256([
                    0x72637735638bb9c5,
                    0xebb32157a8a94c17,
                    0xf9e9ce1126173ef4,
                    0x05a8eab2c64aad29,
                ])),
                Felt::new(BigInteger256([
                    0x9b1213756ab8a41d,
                    0x5926f5f008e7987b,
                    0x62d037187c82c9b6,
                    0x281a4b1c95904cd6,
                ])),
                Felt::new(BigInteger256([
                    0xbae6f8e2662b8e55,
                    0x400d1a9d5e12b866,
                    0x95a212ab06138bf9,
                    0x1e2b9ad49909a627,
                ])),
                Felt::new(BigInteger256([
                    0xebe97f111816dc0d,
                    0xba19d40eb8a05dfe,
                    0x5eb7adf10fdd0158,
                    0x1e01c17f0b6da8fd,
                ])),
                Felt::new(BigInteger256([
                    0x869549087432736d,
                    0x27acac83c991df5e,
                    0xcabea1439bbe8747,
                    0x2c6106b5f966e566,
                ])),
                Felt::new(BigInteger256([
                    0xa7642252d118c6c7,
                    0xfffb78ad6b349275,
                    0x679e876a8ad38ebd,
                    0x1b0151e9597fd0cb,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x283bb7f56ea9871c,
                    0xe17d311886f265c5,
                    0xfd803aa3e58be013,
                    0x3fa3cd0cf5149385,
                ])),
                Felt::new(BigInteger256([
                    0x2371d86be1cea8b4,
                    0x18baa5feb71c938b,
                    0xd4de04d28ace8052,
                    0x0581186d5cc30df3,
                ])),
                Felt::new(BigInteger256([
                    0x43cc2ff124666f8e,
                    0x09485c100acababf,
                    0x93a5d30e3b4a572e,
                    0x2dce80e37f98d9ab,
                ])),
                Felt::new(BigInteger256([
                    0xb37ca73cee782c46,
                    0x012951c4ac722019,
                    0x394ee2b9dc4c9963,
                    0x3b768ee8c8af3780,
                ])),
                Felt::new(BigInteger256([
                    0x3af69833e3cb8d5f,
                    0x54accc445ea3810a,
                    0x9ea9f345cbb28e8f,
                    0x27a9a3f40cbf5a18,
                ])),
                Felt::new(BigInteger256([
                    0x5371d07330f9f64a,
                    0x632f068a614b8a39,
                    0x91bc8882bc33aabf,
                    0x02f3167db078c823,
                ])),
                Felt::new(BigInteger256([
                    0x987a0a41aeac0a22,
                    0x26127298738a6271,
                    0xa42fdaab96f65c58,
                    0x3bd3cc796304317c,
                ])),
                Felt::new(BigInteger256([
                    0x78ba5882bf8cf961,
                    0x52c903fdf4f85433,
                    0x26d68e0d5ef2abdc,
                    0x25ad6c3dafbb8478,
                ])),
                Felt::new(BigInteger256([
                    0x54d176e9ec5f6224,
                    0x071f1f57b9aa11e9,
                    0x3617bea4e41639dd,
                    0x3636fc4b09ae2af4,
                ])),
                Felt::new(BigInteger256([
                    0x9a361d1726dca72f,
                    0x7561e3b14c15c5e0,
                    0xc51f2ecbbe97114c,
                    0x0d1333cdad1fa08d,
                ])),
                Felt::new(BigInteger256([
                    0x892c3d6113159d17,
                    0x4317584b70c79e3f,
                    0x00bd9b7c94761656,
                    0x1d60c44e2fc91495,
                ])),
                Felt::new(BigInteger256([
                    0x4a8c54d226eeae9f,
                    0x069e128e79cb77b4,
                    0x88eb6d482f65e823,
                    0x2ce03eee5ba27be6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7d08f79765568ca3,
                    0xf37aa17abafe5877,
                    0x292b746ecbb4faed,
                    0x1c28fac53ebfff26,
                ])),
                Felt::new(BigInteger256([
                    0xaf1937488d717e1e,
                    0x40aa4f5fff162cc5,
                    0x2cdf22640ab46cf4,
                    0x26d270a5376d00ed,
                ])),
                Felt::new(BigInteger256([
                    0xc2ecddedd7c8f6cd,
                    0xca9871a1e84e2fcb,
                    0x08a0d15256bce848,
                    0x3415a491f59b3073,
                ])),
                Felt::new(BigInteger256([
                    0x255765e69febf706,
                    0x8eb58fe0bd6948de,
                    0xc0ca1269d8d7d2c1,
                    0x317e01afea78b36c,
                ])),
                Felt::new(BigInteger256([
                    0x5c1b20f992901931,
                    0x8788de5b8a7ea3fa,
                    0x31a7981d93086819,
                    0x0d5572a2dd6a3fc4,
                ])),
                Felt::new(BigInteger256([
                    0xae47b7870901a062,
                    0xbe2915d001a23d3e,
                    0x4ebefef48ce232c5,
                    0x2651bd92c0a392ca,
                ])),
                Felt::new(BigInteger256([
                    0xfd36336c3f69593b,
                    0xe624feec8a55171a,
                    0xf9a5988708d50832,
                    0x0757d6b80f7a8d3e,
                ])),
                Felt::new(BigInteger256([
                    0xc23ebb7430b71c95,
                    0x191681f6b453a879,
                    0x47ac7b1c48de5e0f,
                    0x2b019128f61a7eff,
                ])),
                Felt::new(BigInteger256([
                    0xe3c5a4a33aae6e5f,
                    0x7f61f5e97d270118,
                    0xc1d7d61875c6707b,
                    0x09579eb1a6582ff3,
                ])),
                Felt::new(BigInteger256([
                    0x4ff579b29062ccce,
                    0xddf85a9bf84024e6,
                    0x1dc05e4cc0e1d75b,
                    0x320f00187fd0396a,
                ])),
                Felt::new(BigInteger256([
                    0x91a6a0f94dd380c8,
                    0xbd5a721ba4e46f26,
                    0xa7b530d10f9acedf,
                    0x117748756e77aed8,
                ])),
                Felt::new(BigInteger256([
                    0xc708efd798b5a4e7,
                    0x732445c86a23bf3f,
                    0x4c46ff5826ae0207,
                    0x24fe83b26ba79efa,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe581e91e84dc4b8b,
                    0x9fc25e2fda06a595,
                    0x61b5e052f790d5a1,
                    0x0f61d308340ec3ca,
                ])),
                Felt::new(BigInteger256([
                    0x85a7469f5fe3d8e9,
                    0x54a902e7181a4e4f,
                    0x4070be745b4b6644,
                    0x041dadf5f08e3752,
                ])),
                Felt::new(BigInteger256([
                    0x8d539ba2f78d2d62,
                    0xa0cedc2773015382,
                    0xc5493dbc64567695,
                    0x11c9dabb7c1463a4,
                ])),
                Felt::new(BigInteger256([
                    0xdb6bb19ff7be9418,
                    0xed671d5899b4c89f,
                    0x13fb49043cb7eae2,
                    0x1d918f774fed1438,
                ])),
                Felt::new(BigInteger256([
                    0x66a304c08dd2794c,
                    0xf089d2edde5c37a3,
                    0xa62aeb7ca1972b32,
                    0x1f2b1e3341ecc4ab,
                ])),
                Felt::new(BigInteger256([
                    0x47f012b5e624c861,
                    0x9de10c77f0f46425,
                    0xee1bd8a94f7e5b5c,
                    0x2af8f94c9a4b22c5,
                ])),
                Felt::new(BigInteger256([
                    0x475f69e5f6577174,
                    0x80e9363e71581b18,
                    0xf2d0ac3c349f385b,
                    0x1e2682467a98fdf5,
                ])),
                Felt::new(BigInteger256([
                    0xe2674b34d2a73035,
                    0xd3001e0cef8a4579,
                    0x42a03faf1ed5ac18,
                    0x08e8946343deb749,
                ])),
                Felt::new(BigInteger256([
                    0xa9b4adb4b68b4470,
                    0x2c247a651f3e2354,
                    0xfeb812b78fc351bd,
                    0x3958b0e66720135a,
                ])),
                Felt::new(BigInteger256([
                    0x0107054646c9c05a,
                    0x4cf659bdfff1c965,
                    0x61fce19800ff0286,
                    0x1c7304323f038365,
                ])),
                Felt::new(BigInteger256([
                    0x1d726eb09b41a9de,
                    0x302e0629cbc01495,
                    0xc5d143076224b4f6,
                    0x076ec0d3c80d0975,
                ])),
                Felt::new(BigInteger256([
                    0x33dbaaa8fab67192,
                    0x7c5188f8c1c90a9e,
                    0xe61fea3ab4e48887,
                    0x05e71d6206919e84,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xe21f77fa2335b5ab,
                0x774c2b7b0dee7613,
                0xb9441d5ab4a20ee6,
                0x06c26675ff3f223e,
            ]))],
            [Felt::new(BigInteger256([
                0x63e0c6956580441b,
                0x9387f564930b2584,
                0xc250ca2510acc33a,
                0x2df3ba7f79695bdb,
            ]))],
            [Felt::new(BigInteger256([
                0x85f8325fd07a22e3,
                0x7d1d47c14e96751e,
                0x8f01a3728977b34a,
                0x3a853f0a0cbebd44,
            ]))],
            [Felt::new(BigInteger256([
                0xe3bfb7f296e5b31d,
                0xc4beec2f3ceec464,
                0xb21628402248492a,
                0x1cc0ff475a23cc9d,
            ]))],
            [Felt::new(BigInteger256([
                0x9a71335257f76783,
                0x580b1e4ae2fc087c,
                0xdffc2a6150a1eea4,
                0x28dd5934de8c6f37,
            ]))],
            [Felt::new(BigInteger256([
                0x31c827d07eb9f62a,
                0x58eb25bd6ccb7bcd,
                0xce7b85faaf98869f,
                0x2f5f90db6a91626c,
            ]))],
            [Felt::new(BigInteger256([
                0x8c56157586215e03,
                0x313a6c100bcc62c1,
                0xab3d4be7b2bad02f,
                0x3ecbc9f14cb21143,
            ]))],
            [Felt::new(BigInteger256([
                0xcf46b9b715bd0763,
                0x643b45424ad5dbf8,
                0xb2c7ffb3cacccdb3,
                0x0bfd4a37b685c10e,
            ]))],
            [Felt::new(BigInteger256([
                0x1351664af954bbc7,
                0x8e4c54a63739657c,
                0x343223c26807b1b7,
                0x3b3f37ac7dbb35ff,
            ]))],
            [Felt::new(BigInteger256([
                0xde18b54222e5eaec,
                0xc5601f9d0ca25e27,
                0xa707cf72d53351bd,
                0x179aa67db9100e01,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 12));
        }
    }
}
