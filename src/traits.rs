use ark_ff::Field;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use unroll::unroll_for_loops;

/// Trait for implementing a Sponge construction.
pub trait Sponge<F: Field> {
    /// Specifies a digest type returned by this hasher.
    type Digest;

    /// Returns a hash of the provided sequence of bytes.
    fn hash(bytes: &[u8]) -> Self::Digest;

    /// Returns a hash of the provided sequence of field elements.
    fn hash_field(elems: &[F]) -> Self::Digest;

    /// Compresses two given digests into one.
    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest;
}

/// Trait for implementing a Jive compression function instantiation.
pub trait Jive<F: Field> {
    /// Compresses the provided field element slice as input by 2.
    ///
    /// The slice must be of the same length than the underlying hash state.
    fn compress(elems: &[F]) -> Vec<F>;

    /// Compresses the provided field element slice as input by a factor k.
    ///
    /// The slice must be of the same length than the underlying hash state.
    fn compress_k(elems: &[F], k: usize) -> Vec<F>;
}

/// An Anemoi instance, defining the Anemoi permutation over a given finite field
/// for a given instance size.
pub trait Anemoi<'a, F: Field> {
    /// Number of columns of this Anemoi instance.
    const NUM_COLUMNS: usize;
    /// Number of rounds of this Anemoi instance.
    const NUM_ROUNDS: usize;

    /// Width of this Anemoi instance. Should always be equal to
    /// twice the number of columns.
    const WIDTH: usize;
    /// The rate of this Anemoi instance when used in Sponge mode.
    const RATE: usize;
    /// The output size of this Anemoi instance, in both Sponge or Jive mode.
    const OUTPUT_SIZE: usize;

    /// The MDS matrix used for this Anemoi instance's linear layer. It is optional
    /// as short instances benefit from a custom low-cost matrix-vector product for the
    /// Anemoi linear layer.
    const MDS: Option<&'a [F]> = None;
    /// The first set of additive round constants (C) used for this Anemoi instance.
    const ARK_C: &'a [F];
    /// The first set of additive round constants (D) used for this Anemoi instance.
    const ARK_D: &'a [F];

    /// The group generator of the underlying field of this Anemoi instance. It is defined
    /// to possibly speed up the MDS layer for small instances.
    const GROUP_GENERATOR: u32;

    /// The alpha exponent used for this Anemoi instance's S-Box layer.
    const ALPHA: u32;
    /// The inv_alpha exponent used for this Anemoi instance's S-Box layer.
    const INV_ALPHA: F;
    /// The beta constant used for this Anemoi instance's S-Box layer.
    const BETA: u32;
    /// The delta constant used for this Anemoi instance's S-Box layer.
    const DELTA: F;
    /// The quadratic factor used for this Anemoi instance's S-Box layer. Binary fields are not
    /// supported here, hence it is always set to 2.
    const QUAD: u32 = 2;

    /// Helper method to possibly speed-up the linear layer.
    /// It is also used by the S-Box layer as `Self::BETA` is defined as the generator.
    fn mul_by_generator(x: &F) -> F {
        match Self::GROUP_GENERATOR {
            2 => x.double(),
            3 => x.double() + x,
            5 => x.double().double() + x,
            7 => (x.double() + x).double() + x,
            9 => x.double().double().double() + x,
            11 => (x.double().double() + x).double() + x,
            13 => ((x.double() + x).double() + x).double() + x,
            15 => x.double().double().double().double() - x,
            17 => x.double().double().double().double() + x,
            _ => F::from(Self::GROUP_GENERATOR as u64) * x,
        }
    }

    /// Helper method to exponentiate by this Anemoi instance's `ALPHA` parameter.
    fn exp_by_alpha(x: F) -> F {
        match Self::ALPHA {
            3 => x.square() * x,
            5 => x.square().square() * x,
            7 => (x.square() * x).square() * x,
            11 => (x.square().square() * x).square() * x,
            13 => ((x.square() * x).square() * x).square() * x,
            17 => x.square().square().square().square() * x,
            _ => x.pow([Self::ALPHA as u64]),
        }
    }

    /// Helper method to exponentiate by this Anemoi instance's `INV_ALPHA` parameter.
    /// It is left to implementors to provide efficient multiplication chains.
    fn exp_by_inv_alpha(x: F) -> F;

    /// Additive Round Constants (ARK) layer.
    #[inline(always)]
    #[unroll_for_loops]
    fn ark_layer(state: &mut [F], round_ctr: usize) {
        debug_assert!(state.len() == Self::WIDTH);
        assert!(round_ctr < Self::NUM_ROUNDS);
        let range = round_ctr * Self::NUM_COLUMNS..(round_ctr + 1) * Self::NUM_COLUMNS;

        let c = &Self::ARK_C[range.clone()];
        let d = &Self::ARK_D[range];

        for i in 0..Self::NUM_COLUMNS {
            state[i] += c[i];
            state[Self::NUM_COLUMNS + i] += d[i];
        }
    }

    /// Linear layer.
    #[inline(always)]
    fn mds_layer(state: &mut [F]) {
        debug_assert!(state.len() == Self::WIDTH);

        // Anemoi MDS matrices for small instances have been chosen
        // to support cheap matrix-vector product.

        match Self::NUM_COLUMNS {
            1 => {
                // The MDS matrix is the identity.

                // PHT layer
                state[1] += state[0];
                state[0] += state[1];
            }
            2 => {
                state[0] += Self::mul_by_generator(&state[1]);
                state[1] += Self::mul_by_generator(&state[0]);

                state[3] += Self::mul_by_generator(&state[2]);
                state[2] += Self::mul_by_generator(&state[3]);
                state.swap(2, 3);

                // PHT layer
                state[2] += state[0];
                state[3] += state[1];

                state[0] += state[2];
                state[1] += state[3];
            }
            3 => {
                Self::mds_internal(&mut state[..Self::NUM_COLUMNS]);
                state[Self::NUM_COLUMNS..].rotate_left(1);
                Self::mds_internal(&mut state[Self::NUM_COLUMNS..]);

                // PHT layer
                state[3] += state[0];
                state[4] += state[1];
                state[5] += state[2];

                state[0] += state[3];
                state[1] += state[4];
                state[2] += state[5];
            }
            4 => {
                Self::mds_internal(&mut state[..Self::NUM_COLUMNS]);
                state[Self::NUM_COLUMNS..].rotate_left(1);
                Self::mds_internal(&mut state[Self::NUM_COLUMNS..]);

                // PHT layer
                state[4] += state[0];
                state[5] += state[1];
                state[6] += state[2];
                state[7] += state[3];

                state[0] += state[4];
                state[1] += state[5];
                state[2] += state[6];
                state[3] += state[7];
            }
            5 => {
                let x = state[..Self::NUM_COLUMNS].to_vec();
                let mut y = state[Self::NUM_COLUMNS..].to_vec();
                y.rotate_left(1);

                let sum_coeffs = x[0] + x[1] + x[2] + x[3] + x[4];
                state[0] = sum_coeffs + x[3] + (x[2] + x[3] + x[4].double()).double();
                state[1] = sum_coeffs + x[4] + (x[3] + x[4] + x[0].double()).double();
                state[2] = sum_coeffs + x[0] + (x[4] + x[0] + x[1].double()).double();
                state[3] = sum_coeffs + x[1] + (x[0] + x[1] + x[2].double()).double();
                state[4] = sum_coeffs + x[2] + (x[1] + x[2] + x[3].double()).double();

                let sum_coeffs = y[0] + y[1] + y[2] + y[3] + y[4];
                state[5] = sum_coeffs + y[3] + (y[2] + y[3] + y[4].double()).double();
                state[6] = sum_coeffs + y[4] + (y[3] + y[4] + y[0].double()).double();
                state[7] = sum_coeffs + y[0] + (y[4] + y[0] + y[1].double()).double();
                state[8] = sum_coeffs + y[1] + (y[0] + y[1] + y[2].double()).double();
                state[9] = sum_coeffs + y[2] + (y[1] + y[2] + y[3].double()).double();

                // PHT layer
                state[5] += state[0];
                state[6] += state[1];
                state[7] += state[2];
                state[8] += state[3];
                state[9] += state[4];

                state[0] += state[5];
                state[1] += state[6];
                state[2] += state[7];
                state[3] += state[8];
                state[4] += state[9];
            }
            6 => {
                let x = state[..Self::NUM_COLUMNS].to_vec();
                let mut y = state[Self::NUM_COLUMNS..].to_vec();
                y.rotate_left(1);

                let sum_coeffs = x[0] + x[1] + x[2] + x[3] + x[4] + x[5];
                state[0] =
                    sum_coeffs + x[3] + x[5] + (x[2] + x[3] + (x[4] + x[5]).double()).double();
                state[1] =
                    sum_coeffs + x[4] + x[0] + (x[3] + x[4] + (x[5] + x[0]).double()).double();
                state[2] =
                    sum_coeffs + x[5] + x[1] + (x[4] + x[5] + (x[0] + x[1]).double()).double();
                state[3] =
                    sum_coeffs + x[0] + x[2] + (x[5] + x[0] + (x[1] + x[2]).double()).double();
                state[4] =
                    sum_coeffs + x[1] + x[3] + (x[0] + x[1] + (x[2] + x[3]).double()).double();
                state[5] =
                    sum_coeffs + x[2] + x[4] + (x[1] + x[2] + (x[3] + x[4]).double()).double();

                let sum_coeffs = y[0] + y[1] + y[2] + y[3] + y[4] + y[5];
                state[6] =
                    sum_coeffs + y[3] + y[5] + (y[2] + y[3] + (y[4] + y[5]).double()).double();
                state[7] =
                    sum_coeffs + y[4] + y[0] + (y[3] + y[4] + (y[5] + y[0]).double()).double();
                state[8] =
                    sum_coeffs + y[5] + y[1] + (y[4] + y[5] + (y[0] + y[1]).double()).double();
                state[9] =
                    sum_coeffs + y[0] + y[2] + (y[5] + y[0] + (y[1] + y[2]).double()).double();
                state[10] =
                    sum_coeffs + y[1] + y[3] + (y[0] + y[1] + (y[2] + y[3]).double()).double();
                state[11] =
                    sum_coeffs + y[2] + y[4] + (y[1] + y[2] + (y[3] + y[4]).double()).double();

                // PHT layer
                state[6] += state[0];
                state[7] += state[1];
                state[8] += state[2];
                state[9] += state[3];
                state[10] += state[4];
                state[11] += state[5];

                state[0] += state[6];
                state[1] += state[7];
                state[2] += state[8];
                state[3] += state[9];
                state[4] += state[10];
                state[5] += state[11];
            }
            _ => {
                let mds = Self::MDS.expect("NO MDS matrix specified for this instance.");
                // Default to naive matrix-vector multiplication
                let mut result = vec![F::zero(); Self::WIDTH];
                for (index, r) in result.iter_mut().enumerate().take(Self::NUM_COLUMNS) {
                    for j in 0..Self::NUM_COLUMNS {
                        *r += mds[index * Self::NUM_COLUMNS + j] * state[j];
                    }
                }

                state[Self::NUM_COLUMNS..].rotate_left(1);
                for (index, r) in result.iter_mut().skip(Self::NUM_COLUMNS).enumerate() {
                    for j in 0..Self::NUM_COLUMNS {
                        *r += mds[index * Self::NUM_COLUMNS + j] * state[Self::NUM_COLUMNS + j];
                    }
                }

                // PHT layer
                for i in 0..Self::NUM_COLUMNS {
                    state[Self::NUM_COLUMNS + i] = result[i] + result[Self::NUM_COLUMNS + i];
                }
                for i in 0..Self::NUM_COLUMNS {
                    state[i] = result[i] + state[Self::NUM_COLUMNS + i];
                }
            }
        }
    }

    /// Utility method for the mds_layer.
    #[inline(always)]
    fn mds_internal(state: &mut [F]) {
        debug_assert!(state.len() == Self::WIDTH);

        match Self::NUM_COLUMNS {
            3 => {
                let tmp = state[0] + Self::mul_by_generator(&state[2]);
                state[2] += state[1];
                state[2] += Self::mul_by_generator(&state[0]);

                state[0] = tmp + state[2];
                state[1] += tmp;
            }
            4 => {
                state[0] += state[1];
                state[2] += state[3];
                state[3] += Self::mul_by_generator(&state[0]);
                state[1] = Self::mul_by_generator(&(state[1] + state[2]));

                state[0] += state[1];
                state[2] += Self::mul_by_generator(&state[3]);
                state[1] += state[2];
                state[3] += state[0];
            }
            _ => (),
        }
    }

    /// The S-Box layer.
    #[inline(always)]
    #[unroll_for_loops]
    fn sbox_layer(state: &mut [F]) {
        debug_assert!(state.len() == Self::WIDTH);

        let mut x = state[..Self::NUM_COLUMNS].to_vec();
        let mut y = state[Self::NUM_COLUMNS..].to_vec();

        x.iter_mut().enumerate().for_each(|(i, t)| {
            let y2 = y[i].square();
            *t -= Self::mul_by_generator(&y2);
        });

        let mut x_alpha_inv = x.clone();
        x_alpha_inv
            .iter_mut()
            .for_each(|t| *t = Self::exp_by_inv_alpha(*t));

        y.iter_mut()
            .enumerate()
            .for_each(|(i, t)| *t -= x_alpha_inv[i]);

        state
            .iter_mut()
            .enumerate()
            .take(Self::NUM_COLUMNS)
            .for_each(|(i, t)| {
                let y2 = y[i].square();
                *t = x[i] + Self::mul_by_generator(&y2) + Self::DELTA;
            });

        state[Self::NUM_COLUMNS..].copy_from_slice(&y);
    }

    /// A full round of a permutation for this Anemoi instance.
    fn round(state: &mut [F], round_ctr: usize) {
        debug_assert!(state.len() == Self::WIDTH);

        Self::ark_layer(state, round_ctr);
        Self::mds_layer(state);
        Self::sbox_layer(state);
    }

    /// An entire permutation for this Anemoi instance.
    fn permutation(state: &mut [F]) {
        debug_assert!(state.len() == Self::WIDTH);

        for i in 0..Self::NUM_ROUNDS {
            Self::round(state, i);
        }

        Self::mds_layer(state)
    }
}
