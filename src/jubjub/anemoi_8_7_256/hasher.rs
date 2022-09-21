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
        // We can output as few as 2 elements while
        // maintaining the targeted security level.
        assert!(k <= NUM_COLUMNS);

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
                0x4fe395112dcc54fb,
                0x15c99918e5413453,
                0xb38613f9449597a8,
                0x47a8313c3a166c40,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xb5ee376715351d93,
                    0x1a21205687059496,
                    0xc6f1959eaa47c663,
                    0x0638b97ae81e36e9,
                ])),
                Felt::new(BigInteger256([
                    0x302c4327193052a9,
                    0x45442a4f1a48fe26,
                    0x6109c488e1d2c121,
                    0x0088fda876f40e2c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x88f3904c056b8eda,
                    0xb225da59cbb8422d,
                    0x72dc7511d793c167,
                    0x0ac7cf688d05ef9e,
                ])),
                Felt::new(BigInteger256([
                    0x83d8e112a31aa1d8,
                    0x2974cb98d5d8d311,
                    0x1891108e9de5324c,
                    0x4c6ca92d381109c4,
                ])),
                Felt::new(BigInteger256([
                    0xbf3e6fcaba8bb23a,
                    0x382a33ac575ea4ac,
                    0x602b422c14c6aacf,
                    0x60658721f1bf9416,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb115c431c31dcd58,
                    0x3dd09f2844506e34,
                    0xe32353dc39f243f7,
                    0x1caf9c252fcb11d4,
                ])),
                Felt::new(BigInteger256([
                    0x9713550234b19604,
                    0x4c46b9c3d2732701,
                    0x04348a3ed0ce030d,
                    0x03811cec185e6a49,
                ])),
                Felt::new(BigInteger256([
                    0xbf13d96ff30ee2f3,
                    0xe11df0525621d314,
                    0x75f2edcfec49ab43,
                    0x146407cb0fb54157,
                ])),
                Felt::new(BigInteger256([
                    0x4e282fd4ebdd6f31,
                    0x0fbda5bc3a5fff77,
                    0x7d0a5b0760ec796d,
                    0x5594418b416eb0a5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd51e3047803d47fd,
                    0x3dc673f8dc9aee0d,
                    0xfb7f1f1b7c53798c,
                    0x1caa83837f17ddee,
                ])),
                Felt::new(BigInteger256([
                    0x7c43785ac3e21448,
                    0xda8c205febcf7ff7,
                    0x757489a9bf4f7229,
                    0x27f288e3453140d7,
                ])),
                Felt::new(BigInteger256([
                    0xa2b62681e24d2c72,
                    0x7704db2f8d097879,
                    0x3e47f35f84063774,
                    0x1d7cafbbbb4e0321,
                ])),
                Felt::new(BigInteger256([
                    0xb8f03172814f9d30,
                    0x949e6681d9427a6d,
                    0x5d45da7792ac5ade,
                    0x183e6c28e816aa55,
                ])),
                Felt::new(BigInteger256([
                    0xf252694e705cd95b,
                    0x50ae7c9bf5c514c1,
                    0xc95da430eca7f90f,
                    0x231fe497d9941542,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x319a2b9ea5c91009,
                    0x857c77d5bd29bb10,
                    0xebeba5b559d10968,
                    0x4672be08f3f72fb2,
                ])),
                Felt::new(BigInteger256([
                    0x38171f1a9096d880,
                    0x376a0df92a191ac9,
                    0xa20a326a006f42df,
                    0x6abb43e4da2484c2,
                ])),
                Felt::new(BigInteger256([
                    0x5d7b291b87582305,
                    0x601b2ec9f9b5d4db,
                    0x2227b4f01c14c92f,
                    0x572541cc8fa70498,
                ])),
                Felt::new(BigInteger256([
                    0x338790a62392fd15,
                    0xe0109908021cb9cb,
                    0xfe21e049c858fda7,
                    0x0e2cbcb3a8564308,
                ])),
                Felt::new(BigInteger256([
                    0xded20c566373deb7,
                    0x11bddf3265ce8e55,
                    0x388649a92412127b,
                    0x59d5c2a4e4194933,
                ])),
                Felt::new(BigInteger256([
                    0x8eb8bd3af9db6c42,
                    0x23774a62e4287875,
                    0x5a31ed3e6858be9a,
                    0x0bdc2eb247eb923a,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xff585f8c8ebddd33,
                    0xe8b7ef9c02bf3ba2,
                    0x2c922427857e3d7f,
                    0x1818f96d67fa2a31,
                ])),
                Felt::new(BigInteger256([
                    0xd26beb91795eb7ed,
                    0x67af5e0ef1d037a0,
                    0xbbc162f4b189ca95,
                    0x2a5f898f53e07fcd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb31cd1303d8b7a03,
                    0xba28256dcf99cd9b,
                    0x56d63fadf6669999,
                    0x45af2a121ff0687c,
                ])),
                Felt::new(BigInteger256([
                    0xd78b711ed97c2862,
                    0x5b30bd5f9e069673,
                    0xed79da84d990e3df,
                    0x65d8b1413623f483,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4cee0effed557e0b,
                    0xac24f80238c352be,
                    0x0c15be3c776bbcca,
                    0x40088ac9d2f66c6e,
                ])),
                Felt::new(BigInteger256([
                    0x43307520ec0041b5,
                    0x7a9649d57797b598,
                    0xc89f9116e6794dbf,
                    0x235e0bb7738bc3de,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x02ca5f598ae821ad,
                    0xc0dc6016dfca11eb,
                    0xfa812f794c41dce4,
                    0x40952537ce84ecf7,
                ])),
                Felt::new(BigInteger256([
                    0x54286137f8a7d5ea,
                    0xd9605f5006842a7b,
                    0x8df337a11e9e71c2,
                    0x4115981d7d048ce5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x874ac94411bdeaa2,
                    0xbefcf68200ea2773,
                    0x207f03b6f42820f6,
                    0x3a214acb020bb7d1,
                ])),
                Felt::new(BigInteger256([
                    0x36659416f7b267c1,
                    0xef53da718ff168aa,
                    0xb27f41c6dedceea5,
                    0x57ec44a31c10e4cc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x83c58586dd6350ed,
                    0x20503430b39b78dc,
                    0xa2f53823e5348025,
                    0x4b6ad201217163c3,
                ])),
                Felt::new(BigInteger256([
                    0x769a2fdbc2f8bb67,
                    0x051921879bcb04c3,
                    0x534245d73aafce0e,
                    0x33bff771fe4ee01c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe15d350ee8130050,
                    0xf2b2754fdf4a6270,
                    0x770d9dea5944188a,
                    0x5ffee5bc7c9c2a99,
                ])),
                Felt::new(BigInteger256([
                    0xbf1b5482ec87a194,
                    0x88d82145e8b26e9d,
                    0x28e33f8fe09629d5,
                    0x61a29a281f835715,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x91cf5abb003f33ae,
                    0x5b038ac56106ba2f,
                    0x640667c53fd1b099,
                    0x487d67f7454ac301,
                ])),
                Felt::new(BigInteger256([
                    0xefd658e9f5e90613,
                    0x506fbdde85733632,
                    0x60fd24faca0e8b79,
                    0x6d0ee88a8fef9916,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcb6ce30e6850fc39,
                    0x9cf452680eb1d899,
                    0x73c479ceee765964,
                    0x6546c21fafaeaab2,
                ])),
                Felt::new(BigInteger256([
                    0x3c0bb4da7dbb613f,
                    0x645c81ed2af038d1,
                    0x907791b4772c6604,
                    0x246eba32ef1e1fbc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x052a21b18ca33a8a,
                    0xb0a57fdb73753afe,
                    0x6efbce811c5e9670,
                    0x42b12b98a3666d98,
                ])),
                Felt::new(BigInteger256([
                    0x3f803b097e153c51,
                    0x9b4a059f00ed0ba0,
                    0x552136463b9691c4,
                    0x15ceaf206a77ba79,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
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
                    0xc7b8dd6cb2edad10,
                    0x8bba42c1be26191e,
                    0xf349cea23c7f0c3a,
                    0x1794a22a64979529,
                ])),
                Felt::new(BigInteger256([
                    0x5906fb290d3bb6a6,
                    0x293cd75c0dc97699,
                    0xb2e5d13aab0e5a7c,
                    0x370bf9791c5e1947,
                ])),
                Felt::new(BigInteger256([
                    0x4546b90af193bfa2,
                    0xb9816ef761dbf25c,
                    0x1b8b88b5c5ccc1ba,
                    0x5513f95355976f3f,
                ])),
                Felt::new(BigInteger256([
                    0x030bc7fd5c4b2690,
                    0x582fdc03c2c8c62a,
                    0x0894484644ebcc11,
                    0x1d746bf26a593d77,
                ])),
                Felt::new(BigInteger256([
                    0x4ab2f883220ecef2,
                    0x731e71e240c6474d,
                    0x629a674b2657cc04,
                    0x5355af6ac1febb46,
                ])),
                Felt::new(BigInteger256([
                    0xc2c4aac6e0b01597,
                    0x8e527d3bbea3bc6d,
                    0xa25d07d5a1f3168f,
                    0x059f43ae7c95b975,
                ])),
                Felt::new(BigInteger256([
                    0xb2a2630060f34269,
                    0x077e7327d137e632,
                    0xc2e88455215a162f,
                    0x39ca5fd2f4063e06,
                ])),
                Felt::new(BigInteger256([
                    0x6504a84f52512b83,
                    0x5bc37e0efb37298b,
                    0x33add1b940d92bf3,
                    0x2bf8efda8b3caacf,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x352fcf9a31e2f3ee,
                    0x1a146aba7e95a310,
                    0x4c77b3e6fcaeb008,
                    0x29a465d703f350d3,
                ])),
                Felt::new(BigInteger256([
                    0xf8d79c624c847783,
                    0x0c5086a411f880c8,
                    0x6c85b060c2e98459,
                    0x4e6e68d42fe4f330,
                ])),
                Felt::new(BigInteger256([
                    0xbd6449d2819ebeae,
                    0x6b7ab65cb8eb4905,
                    0xe5d0019751e02a79,
                    0x31014dd2093dbb6e,
                ])),
                Felt::new(BigInteger256([
                    0x4262825f4f20e264,
                    0x6d9f749438eeeef4,
                    0x23423ec3430c6ede,
                    0x3745c7f238c3f6b0,
                ])),
                Felt::new(BigInteger256([
                    0xedab7b09c2a6c7a9,
                    0xf2dd6f6a7adef2e4,
                    0xb333eb51713ff67a,
                    0x14fe574b6afc92f4,
                ])),
                Felt::new(BigInteger256([
                    0x4c92938daa29b533,
                    0x0012feb1c273240b,
                    0x1028e72dbda16e9b,
                    0x0c129506783c8478,
                ])),
                Felt::new(BigInteger256([
                    0x8add14d43ee2f35a,
                    0x05011d29f9c93a05,
                    0x4aadb5ef7add3926,
                    0x58c9c962314b60f7,
                ])),
                Felt::new(BigInteger256([
                    0x2e445656a5ff78e4,
                    0xb2d0a368ee966358,
                    0x0eea908be5a49f22,
                    0x3304f1ac4cc250b4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf2b08c5d1c34ae49,
                    0xb068a42c97139ce3,
                    0x6ef2050ce7a8d83d,
                    0x1a2f1d0f3428d0ca,
                ])),
                Felt::new(BigInteger256([
                    0xf68bf9b407509fe0,
                    0x8bc93d63ba31373d,
                    0x45d48ec97c7acaf7,
                    0x1118c8bd0f09655f,
                ])),
                Felt::new(BigInteger256([
                    0x99fec4a6112584d7,
                    0x76af4d7b561c11bc,
                    0x474fd6a5867151aa,
                    0x65c37310b9f06ae7,
                ])),
                Felt::new(BigInteger256([
                    0x86268716e660859d,
                    0x91bd200e7151fb07,
                    0x1cf78dfd2756c7c1,
                    0x0871c911e8e536de,
                ])),
                Felt::new(BigInteger256([
                    0x7e961bedc10fd016,
                    0x6e4eaafa47f7281f,
                    0x28d0b537665649f7,
                    0x0907ac9bb4dfedd0,
                ])),
                Felt::new(BigInteger256([
                    0xea5dc31f96109889,
                    0x9189c36bdd258738,
                    0xa99e5f10b6d60ae4,
                    0x3992366201e2bb30,
                ])),
                Felt::new(BigInteger256([
                    0x32c4eca3bddcd6ab,
                    0x2c9414e2cda2e49b,
                    0x5c2143e855c71272,
                    0x4fe81bd395a2842e,
                ])),
                Felt::new(BigInteger256([
                    0x41660898701d161b,
                    0x14f8cd105fdfeeec,
                    0xb4dba56b1fdac5cd,
                    0x5fb770e50d71faa1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7065208ca5c2b05b,
                    0x0dfac7a4576f8c3b,
                    0x0f04b63c6b98d108,
                    0x4f065ec4d41566e2,
                ])),
                Felt::new(BigInteger256([
                    0x586c5273b605e8de,
                    0x97501bb12cfd9c09,
                    0x9a4077dccca5df5f,
                    0x123d370b37e23b33,
                ])),
                Felt::new(BigInteger256([
                    0x7956622e5f7e9d2f,
                    0xc685ae5f57ca1ae4,
                    0xda358b3e444e29de,
                    0x443b8b9e0f2a77fa,
                ])),
                Felt::new(BigInteger256([
                    0x6f7ecbb13003a847,
                    0x26ebb846690207c1,
                    0xe90970464105679d,
                    0x36611474d43c486d,
                ])),
                Felt::new(BigInteger256([
                    0x768bd076e16df04e,
                    0x531d66822930d2dd,
                    0x20e868a338959937,
                    0x6369b07e6f620614,
                ])),
                Felt::new(BigInteger256([
                    0x7633fb2b41d4650e,
                    0x197f3bf19d80afcb,
                    0x29f9f9425c9b7036,
                    0x02e19f6325dbaa8a,
                ])),
                Felt::new(BigInteger256([
                    0x3d93e2ed44fe3e87,
                    0x4996e91b5bfee80e,
                    0x747949104a98bb84,
                    0x2d9fb03a34ebc7da,
                ])),
                Felt::new(BigInteger256([
                    0x664cf7f063ad9212,
                    0xd9313779ad0a21c6,
                    0xc796d5fb624a7053,
                    0x1091b49cc4f4c4f3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaf8927375a7dff3d,
                    0xa92f0d7d371d5ca1,
                    0x6b0fc4acd0975311,
                    0x16e71ac69e7d9b9a,
                ])),
                Felt::new(BigInteger256([
                    0x1e7eade5a37304db,
                    0x68b2f9c713b20777,
                    0x896765c7cc7d2aa2,
                    0x0eeccb144459cf00,
                ])),
                Felt::new(BigInteger256([
                    0xd62ee6a844a2f40b,
                    0xafecaa3955f1eb83,
                    0x9472c59b35d9b151,
                    0x2c6e58a3ed36a54e,
                ])),
                Felt::new(BigInteger256([
                    0xf327762e73d897a1,
                    0xbea4dbc9278817d7,
                    0xba47f014dbf02313,
                    0x5769b4ca94c2b4d2,
                ])),
                Felt::new(BigInteger256([
                    0x7224b8c8486c2e03,
                    0x9a74e98c27574daa,
                    0xfd4d7330f8f3ffb3,
                    0x1a1e8634b45b7fda,
                ])),
                Felt::new(BigInteger256([
                    0x50b91210b422188c,
                    0x1adb13734354ec27,
                    0x384980730b69aec4,
                    0x3a2ad1aba5bbce56,
                ])),
                Felt::new(BigInteger256([
                    0x45f94eff549d583f,
                    0x5d8ed9e2e38a8faf,
                    0xb68f29a058492b16,
                    0x40a66cceed32d067,
                ])),
                Felt::new(BigInteger256([
                    0x00f07666315d88b1,
                    0x569d8e705686fa5a,
                    0xf8ed0c860a361d4e,
                    0x6f9fe98accc07ee2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd94f6c87f1d7d481,
                    0xea755c7c28c1ca18,
                    0x4a3eb2016ea3eb9a,
                    0x0514699d7e3fc07c,
                ])),
                Felt::new(BigInteger256([
                    0x6f09bddafcc519e5,
                    0xbb428ed0b0e22690,
                    0x20f7fee6666b61d3,
                    0x5329da23cb521599,
                ])),
                Felt::new(BigInteger256([
                    0x39ff474c644c64a1,
                    0xd3a14bec5eca073c,
                    0x4acd4391f886108f,
                    0x32b9341c02f786cf,
                ])),
                Felt::new(BigInteger256([
                    0xadeb9f0cf941084e,
                    0x101edb92b488e0f6,
                    0xd11ce8be26c56f90,
                    0x1834577d7e7b9b4a,
                ])),
                Felt::new(BigInteger256([
                    0x7395c01b791ffb79,
                    0x1a5fe51a1e02d9d6,
                    0x59fe10b06debbc60,
                    0x41e431ca0d67b23d,
                ])),
                Felt::new(BigInteger256([
                    0xeadcdd093b6f7d4c,
                    0x6ee61f38213e2752,
                    0x97dbe9c0c99f45b8,
                    0x50d8f5eaf8b1e6b7,
                ])),
                Felt::new(BigInteger256([
                    0x929baf1397809192,
                    0xc9f1ed30d558631b,
                    0xb02e5a95ae2300a0,
                    0x546aabdcc04d9727,
                ])),
                Felt::new(BigInteger256([
                    0x2f1da4d6b6d35e63,
                    0xd714d78385539377,
                    0x3b1dbf5ca39ab145,
                    0x0fd04bbe9e303477,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x3cb36ee728c18e63,
                    0x4c4b347f74fe72a6,
                    0x53c128bec0066a45,
                    0x7229c39c0a7ce4a2,
                ])),
                Felt::new(BigInteger256([
                    0xfb637e6c65a2eb00,
                    0xeee2c56857d56c4a,
                    0xe8b5750ee4681929,
                    0x11513a57a46dbf1e,
                ])),
                Felt::new(BigInteger256([
                    0xbf099b3295958548,
                    0x9b48da13062f061e,
                    0x2ee6377a3c3ad32d,
                    0x0657e8fab101a440,
                ])),
                Felt::new(BigInteger256([
                    0x11a43f9ac3f10b16,
                    0xeb61ee4c02bb042f,
                    0x7436b80f5e89ece9,
                    0x267e0ffdc8bc6c77,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd73812d928be2e2f,
                    0x7dd17cadf6964232,
                    0xc492153613bd7f76,
                    0x0402220e3cf156b9,
                ])),
                Felt::new(BigInteger256([
                    0x0ea4300628794ceb,
                    0x31c47c02b4d64a13,
                    0xf90e99d92e3f4322,
                    0x2220bad5a7293712,
                ])),
                Felt::new(BigInteger256([
                    0xd5bd20cb7b76c8e5,
                    0xad3fbfa8db9bab1c,
                    0x42d63657d91ebba5,
                    0x3e6aaf85036794da,
                ])),
                Felt::new(BigInteger256([
                    0x27e9df17cdc12bd4,
                    0x82727125af28253d,
                    0xbcbd528d0e06a035,
                    0x68bc2e2c047ba58c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x988cbe495eed157e,
                    0xf8e39f62385cd85b,
                    0x6c54e377857e6a40,
                    0x65ae21b08c02cbd7,
                ])),
                Felt::new(BigInteger256([
                    0xec6ab1db94fe73e2,
                    0x5209635ecbf16dca,
                    0x76bb3c3e7378edde,
                    0x1a08dc43af239846,
                ])),
                Felt::new(BigInteger256([
                    0x19848419815fba88,
                    0x25423c824487b7d1,
                    0x33838e327bccc05a,
                    0x05c665f951df6f30,
                ])),
                Felt::new(BigInteger256([
                    0xf2e19dd1c9468417,
                    0x2700e8ca639914f5,
                    0x09b7060536699f69,
                    0x26f9a533963dfa38,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x568b5bb8ba3b36c2,
                    0x823090c2236bbb7d,
                    0xa60f97ddcdcb3ac5,
                    0x0e0d9c076da7562f,
                ])),
                Felt::new(BigInteger256([
                    0xf7e905dcd9a5c1f9,
                    0x6a352cd8232916bd,
                    0x3f80d0adfff23d4a,
                    0x4e356be74344b20f,
                ])),
                Felt::new(BigInteger256([
                    0x2d1450a40efd28a2,
                    0x0a9e14a90bfd6deb,
                    0x669a985cd5792671,
                    0x5d66ee561b68f2cb,
                ])),
                Felt::new(BigInteger256([
                    0x78a5d3c21964a865,
                    0x35e3e99fce959e23,
                    0xd7d5157df10a0c63,
                    0x67b464989b9911e4,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xde774257db85c7a3,
                    0xe9ea576dea88da89,
                    0x2407e087aa44815b,
                    0x2a9460ffc98676c9,
                ])),
                Felt::new(BigInteger256([
                    0x7083cd7ec9210794,
                    0xc6df914fe7bac4ca,
                    0xcc90ab73b2da74af,
                    0x320ab3b822dd6b82,
                ])),
                Felt::new(BigInteger256([
                    0xdc1302786da4e07b,
                    0x7ee8d7e8eb2d08a6,
                    0x6297b31590c6959a,
                    0x207bf1b200b609ae,
                ])),
                Felt::new(BigInteger256([
                    0xb60bbda1ce858898,
                    0x9cbed66d7986921a,
                    0xd9a5db183b2b0a13,
                    0x373b16a005d58265,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x33ed384d7fe1ed53,
                    0x6aa022ab609b4bf5,
                    0x72347ef8a428f8c6,
                    0x4f42da11dffdf615,
                ])),
                Felt::new(BigInteger256([
                    0x0e5a670455bc5f29,
                    0x764d33cfa462c5df,
                    0x807b92c2bd0c6ee6,
                    0x10acb3e18f087067,
                ])),
                Felt::new(BigInteger256([
                    0x61d8f55fc1bfc310,
                    0x796cba6d02716ae7,
                    0x75b45dd3f4fb93ea,
                    0x236d0a34d147e687,
                ])),
                Felt::new(BigInteger256([
                    0x6288c539eca0336e,
                    0xee03c52873bb4542,
                    0x9f300e1083b5ab53,
                    0x5178055fc6633392,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa5442fe944fe4f22,
                    0x0506cac1f4096abf,
                    0x41affa7b7cf6716d,
                    0x4ce0cf59dcc6fa45,
                ])),
                Felt::new(BigInteger256([
                    0x73568a49d06b045f,
                    0xa976535a9a818121,
                    0xbf751d8c7168473f,
                    0x722cfc615af5a930,
                ])),
                Felt::new(BigInteger256([
                    0x11c833ae2a23584f,
                    0xc6506821ae1cbe8a,
                    0x668ed03758de2347,
                    0x28b69e3a7626c793,
                ])),
                Felt::new(BigInteger256([
                    0xc115f3768fa6710b,
                    0x234a55729c25cca0,
                    0x25894ba2eb063f9b,
                    0x5c1885a5b1996143,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1bf618e38b45b275,
                    0xaead12aee4fcfdf6,
                    0xa383e1bfeab704d9,
                    0x2c0fec9b689d1d86,
                ])),
                Felt::new(BigInteger256([
                    0xcc632089ed38ed67,
                    0x03b65af1e0c6cd26,
                    0x522d8d9114830865,
                    0x089182aeee24e3d8,
                ])),
                Felt::new(BigInteger256([
                    0x811a6527c617cbd6,
                    0xa5193ee2ff900eae,
                    0x0e5212bde77fb791,
                    0x57ca19fbeacbec69,
                ])),
                Felt::new(BigInteger256([
                    0xdcefbf28ed5ad789,
                    0x5d0ccbdd81c86c98,
                    0xf37c7020fb214fba,
                    0x6c4ee3c378f2bff5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xdcd51d63aee4f9c1,
                    0x0d5a73bcb9225469,
                    0x964173453252da77,
                    0x3565d7ee02c4ac83,
                ])),
                Felt::new(BigInteger256([
                    0x658cbc87a6d548a2,
                    0xdb06739dd9820619,
                    0x2cdac04f8661ba0d,
                    0x1e1c573630bcd459,
                ])),
                Felt::new(BigInteger256([
                    0x08f0206ed3cc2ccc,
                    0xb22d144e82c79e7e,
                    0xae565854d742a3d9,
                    0x6c26e067f9e26db9,
                ])),
                Felt::new(BigInteger256([
                    0x819d12414cd8f35a,
                    0x676ffdc82f536dc2,
                    0x87b05c3f8eb3dadd,
                    0x1249c1de8ab65d1a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x454181d3a9d9a764,
                    0x097a7d41af86befa,
                    0xbb13b5be2884bd31,
                    0x5f6c9a758f2b5b74,
                ])),
                Felt::new(BigInteger256([
                    0x85daf402a82aa775,
                    0x33b16338b2a197d2,
                    0x552f3e8e0f41a351,
                    0x3c6f91ce05220f13,
                ])),
                Felt::new(BigInteger256([
                    0xb9e2ee6ba77e2c9d,
                    0x132f509fe1af8b6b,
                    0xda798df25deb087e,
                    0x31471c0ec7de611f,
                ])),
                Felt::new(BigInteger256([
                    0x4ee661c04a410147,
                    0x193f639099dc18f9,
                    0x4b7fe96ad36dfb9b,
                    0x1aae0e9ebcee75e3,
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
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
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
                    0xc211d17423dc0bcd,
                    0x01fd308f4c558abb,
                    0x82ada6f86c0736a6,
                    0x11ce51d800b9496a,
                ])),
                Felt::new(BigInteger256([
                    0xab342e25d68a3265,
                    0xe065d60102164820,
                    0xec20dd9c6d8597fc,
                    0x4b235e1b94123a8f,
                ])),
                Felt::new(BigInteger256([
                    0xf74432b194f05bf1,
                    0xbe984546e77d5d1f,
                    0xdd1912740c9ff4bd,
                    0x70b02b152539c192,
                ])),
                Felt::new(BigInteger256([
                    0x023fc34cd0c6e66e,
                    0x6d57c241c53fb0af,
                    0xcff2b1213e49949e,
                    0x1741f286f90dea19,
                ])),
                Felt::new(BigInteger256([
                    0xf553341bf2bcbbcb,
                    0xe09b1397babd4df7,
                    0xbb5775665890a931,
                    0x4a86fbb21b1815ea,
                ])),
                Felt::new(BigInteger256([
                    0x2adc4c3c6186a98a,
                    0x148f434b8b0cd440,
                    0x27265603c9f9220e,
                    0x3fffef2e9804cc51,
                ])),
                Felt::new(BigInteger256([
                    0xf03c7441ecfd4e09,
                    0x2a48b2aff27de1e4,
                    0x9eeaba326685b39d,
                    0x4797ec0f4969973b,
                ])),
                Felt::new(BigInteger256([
                    0xce8788f2781f8d94,
                    0x9c94c659905059cd,
                    0xdafaeaa291447bb9,
                    0x6015f3880a39e0fc,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x116fceccd9398388,
                    0xee7446bf32a65d5b,
                    0x6135031ba8401a28,
                    0x2bd5eee50f20371f,
                ])),
                Felt::new(BigInteger256([
                    0xf921329e91f5ff7f,
                    0x1e28b3d37367ce98,
                    0x1d33058aa84a9498,
                    0x126bb943f56524f8,
                ])),
                Felt::new(BigInteger256([
                    0x7729e1ecbfa5b113,
                    0xb4f016400b2664c8,
                    0xf65f482b33fd9b4a,
                    0x3f61e26374ff7713,
                ])),
                Felt::new(BigInteger256([
                    0x39246da64e5e2afa,
                    0x83cea7e8a2416ef3,
                    0x00db2a0cd828d1aa,
                    0x18adf6e4ce8e4b59,
                ])),
                Felt::new(BigInteger256([
                    0x593b8faf2a5b1a8b,
                    0x8479627dafcab6a4,
                    0x033fc2487ce8572a,
                    0x385a6bef262f0196,
                ])),
                Felt::new(BigInteger256([
                    0x4091aa8b281143c4,
                    0xb0a19f6602a08d72,
                    0x4397416944c439a8,
                    0x1ede73a6acc7e14d,
                ])),
                Felt::new(BigInteger256([
                    0x3cc4afe34bc131f2,
                    0xea76fdd6a1a7a9cc,
                    0x9cfe39f2d03d2a87,
                    0x33ce4e68334176fa,
                ])),
                Felt::new(BigInteger256([
                    0x5d9097492db7d624,
                    0x151a7ad1afcec0e2,
                    0xe343d236d9d67cae,
                    0x3dc65b5d05d4f589,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe3e79509d2dec114,
                    0x9feca77aff99944e,
                    0xa4d978ad2026f05f,
                    0x08cdefab0e92468c,
                ])),
                Felt::new(BigInteger256([
                    0xff4007329f1b6c5e,
                    0x6a0a3ac6ff7d3c40,
                    0xa1a713ffba5d74f7,
                    0x6343345baa23e54f,
                ])),
                Felt::new(BigInteger256([
                    0x89cf7bafb5bf341e,
                    0x4daca9e7b57e4bc2,
                    0xde8f4d1dca9718cb,
                    0x5338d0131a3999ad,
                ])),
                Felt::new(BigInteger256([
                    0x6dfbfd79eaab7ce1,
                    0x574f9dd6624ccd11,
                    0x6b4611dde9d80f12,
                    0x221c392cee330852,
                ])),
                Felt::new(BigInteger256([
                    0x45dfe7e02d37b4d9,
                    0x5beef69ed926c0a5,
                    0xeec3576892bd3ef6,
                    0x01eba451238fc7cc,
                ])),
                Felt::new(BigInteger256([
                    0xa77938a8282e2565,
                    0x4eee6ff8adae13d8,
                    0xe77bb4ca5718a594,
                    0x44b9fa0dce1be598,
                ])),
                Felt::new(BigInteger256([
                    0x61e8f45f93430389,
                    0x715fbaf2ea4acdfc,
                    0x98837965c56cb76f,
                    0x0f6fb13490d2c390,
                ])),
                Felt::new(BigInteger256([
                    0xf395069883083101,
                    0xbe0ea53e5ba5d6fc,
                    0xe4f3bd096c3e6fab,
                    0x649fec372117a3db,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xea71f3a7131766c4,
                    0x23257341ff726874,
                    0xc86b2613794bf36f,
                    0x3cf434b515e08b1b,
                ])),
                Felt::new(BigInteger256([
                    0x6c59feed5e1dfe80,
                    0x3d63267bf79a2cc0,
                    0x932a44ffb5a5d05d,
                    0x4fbfaf029826e4b4,
                ])),
                Felt::new(BigInteger256([
                    0x144b715302d8d5fe,
                    0x2368c3d4dd27d5e8,
                    0x3863093bb62f6945,
                    0x3c1bfdc76534d771,
                ])),
                Felt::new(BigInteger256([
                    0x6f8fd6b2f52e215f,
                    0xcca568d4be12e7b0,
                    0x576a856ee9c8e08d,
                    0x677d34d041ddb882,
                ])),
                Felt::new(BigInteger256([
                    0x46ff6e387cf00ca5,
                    0xac38bd3765cec49b,
                    0xeee9ae7f963212f4,
                    0x610d4593d48617dd,
                ])),
                Felt::new(BigInteger256([
                    0x4287a1850bc23dc7,
                    0xbab90611520890ac,
                    0xf206e99f346af388,
                    0x058d4c99c01c9da3,
                ])),
                Felt::new(BigInteger256([
                    0xfb6d769c7aed5529,
                    0xe6281c8cc01f1188,
                    0xe8b3c86733cc301e,
                    0x21925fbab676dc72,
                ])),
                Felt::new(BigInteger256([
                    0xda068aeb78a69fed,
                    0x7e17b0d3be4dc231,
                    0x11b3ee072deba4e8,
                    0x4f39e10bb3acff6a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd310e1e4743e20d5,
                    0xb7657aa2692added,
                    0x4e1233b837e1f501,
                    0x23fb088abfcdc18a,
                ])),
                Felt::new(BigInteger256([
                    0x5ec7aa97a60e4495,
                    0x6b9f8ac08d8bcdcd,
                    0x321cd720027c2e0e,
                    0x6f101b0b7aa25bbf,
                ])),
                Felt::new(BigInteger256([
                    0x5ce8d4912135db35,
                    0x307e72acc38f9a37,
                    0x71d295f2eaaa66b4,
                    0x682b065ab8ac813c,
                ])),
                Felt::new(BigInteger256([
                    0x460241b6301d0bed,
                    0x2a6644eacac1b059,
                    0x01c7b3cb4bee7786,
                    0x3322517c110d4e7b,
                ])),
                Felt::new(BigInteger256([
                    0x9bd4a3e94da865d1,
                    0x868e8cbf1097c556,
                    0x322349eaad38518e,
                    0x69963446d8dc9c54,
                ])),
                Felt::new(BigInteger256([
                    0x4854105331f65cbe,
                    0x3f6bc78be9c247df,
                    0xe948c48b89111c7b,
                    0x31d6b4816e5c358b,
                ])),
                Felt::new(BigInteger256([
                    0xa776d0e46a085a01,
                    0xf921ad68b2f511db,
                    0x99b45d6ec5ffc9b1,
                    0x4eb6f0b056c21244,
                ])),
                Felt::new(BigInteger256([
                    0xa5694181c18206f4,
                    0xcef8d6e0ebde96cc,
                    0xd136a126f93765a7,
                    0x6f4b260be2def33f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd278f179ca0b383b,
                    0x49005617e664ebd4,
                    0x4f87064157e62115,
                    0x21b19fa795dbc5f5,
                ])),
                Felt::new(BigInteger256([
                    0x8c29800d0d2e6038,
                    0xa48cadea2eca2f1a,
                    0x0b69b59d872bd4d2,
                    0x3e0d8be88bbd8cd5,
                ])),
                Felt::new(BigInteger256([
                    0xf0d1cee83fc9bb54,
                    0x689f7243a8968c99,
                    0x8021b6bed9c39ca7,
                    0x45b97668efcc249b,
                ])),
                Felt::new(BigInteger256([
                    0xc0b0b18ee8effd51,
                    0xb2b11638a871fa90,
                    0x00dd3a7bf72395b1,
                    0x3979112659a2d163,
                ])),
                Felt::new(BigInteger256([
                    0x24ad67a8ea1a4760,
                    0x62448d74cb045248,
                    0x656aad26fdac5a57,
                    0x18ddc985b543bc19,
                ])),
                Felt::new(BigInteger256([
                    0x8b21599ca4bec6ac,
                    0x133bd3c98530a98b,
                    0xd41c2bfae56a7bc9,
                    0x30bb220d44e171b7,
                ])),
                Felt::new(BigInteger256([
                    0x2feb40b553013459,
                    0xb66d738d313d1870,
                    0x510e1e282619b777,
                    0x3c21c50673b426f9,
                ])),
                Felt::new(BigInteger256([
                    0xad79b6fff6a269c7,
                    0xcce24fb8c1fa5eaa,
                    0xbcd17d29613450eb,
                    0x201279538cc71f29,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xfbbd0a1abe5713aa,
                    0x93d66a8f7b2f1cc5,
                    0x4f6d8830f29f656d,
                    0x0494054391e10b9a,
                ])),
                Felt::new(BigInteger256([
                    0x0d07be072993f616,
                    0xda44b3b45a90707a,
                    0x5cec2d1e42f20613,
                    0x37cf4a556d2a2b96,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xacf533a4a434f714,
                    0x2b113c56d231ed4f,
                    0x07684b8decdc3b1c,
                    0x426cd1934058eb94,
                ])),
                Felt::new(BigInteger256([
                    0x368e0f1ef63a78be,
                    0x6079492564001351,
                    0x8292145e32a40b52,
                    0x16ef41ae82075f57,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb2114262e04cd006,
                    0x1e25dbe47ce4902c,
                    0x9fd871aa014b2a9b,
                    0x6b7487a9dde23b07,
                ])),
                Felt::new(BigInteger256([
                    0xdf4c4fad5e44f7f9,
                    0x790a4c292f8a82c0,
                    0x80724243a9e28d47,
                    0x410281774561927e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x839fac5cc9385f64,
                    0x8ccea56b2f692968,
                    0x0caa303aa3446136,
                    0x6b748a5d891048fb,
                ])),
                Felt::new(BigInteger256([
                    0x708ed99ff30a6a5d,
                    0x4c5b7274f1c058e2,
                    0xe41c0e23e75a71a8,
                    0x41fc292cb54046ab,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x34bfc6a280665529,
                    0x7333928e649640fe,
                    0xf8ef2c80d8f784b2,
                    0x674cb7a1e449c145,
                ])),
                Felt::new(BigInteger256([
                    0x69d3705264d8f076,
                    0xa69b7382176bf4f4,
                    0xe1203199ed380487,
                    0x59fd9b470670006b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x82f865ad32712d9e,
                    0x525b76d916122496,
                    0x6eff113e73d1cbc9,
                    0x1758f788424d59bd,
                ])),
                Felt::new(BigInteger256([
                    0x0aec2863b3cc166e,
                    0x4dd76d3264740a62,
                    0x12a841761756f78c,
                    0x1531ad0ac604bddc,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcfbc5e9a936c62eb,
                    0x4b1480d8c07b5811,
                    0x35d0c57797df5ed3,
                    0x58f925630659a1cb,
                ])),
                Felt::new(BigInteger256([
                    0xa13a397882e4e03c,
                    0xd37151b46479657b,
                    0xd176c8ff17609ed0,
                    0x5e1d55b479769e01,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x9fbe75fec78dbd67,
                    0x7fee219f4cd0a03f,
                    0xc0fcf4909c331340,
                    0x494116b31877acc2,
                ])),
                Felt::new(BigInteger256([
                    0x2174081afdd039c0,
                    0x35979eb88175bfcd,
                    0x9cd937f195c47dc8,
                    0x3f4926550649f127,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x37db5722c7dbbe7f,
                    0x5fa3f15c5c976dfd,
                    0x08504b4c7df784d1,
                    0x677c7997470ce72c,
                ])),
                Felt::new(BigInteger256([
                    0xebce6715a2cdf669,
                    0xefb355a8999c357d,
                    0x22c761042f1fb1a5,
                    0x271526d434c36b61,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf5fcdbd2690be51b,
                    0x0f21646ec70b3fe9,
                    0x320d1e5fbce7ae5f,
                    0x523faaef544b90a3,
                ])),
                Felt::new(BigInteger256([
                    0x55da19f11aea992e,
                    0x2e1210f1ca959626,
                    0xc20aef6547209a81,
                    0x5cbf795359b1ad66,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
