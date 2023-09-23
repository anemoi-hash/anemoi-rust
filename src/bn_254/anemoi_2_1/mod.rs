//! Implementation of the Anemoi permutation

use super::{sbox, Felt, MontFp};
use crate::{Anemoi, Jive, Sponge};
use ark_ff::{One, Zero};
/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over BN_254 basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiBn254_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiBn254_2_1 {
    const NUM_COLUMNS: usize = NUM_COLUMNS;
    const NUM_ROUNDS: usize = NUM_HASH_ROUNDS;

    const WIDTH: usize = STATE_WIDTH;
    const RATE: usize = RATE_WIDTH;
    const OUTPUT_SIZE: usize = DIGEST_SIZE;

    const ARK_C: &'a [Felt] = &round_constants::C;
    const ARK_D: &'a [Felt] = &round_constants::D;

    const GROUP_GENERATOR: u32 = sbox::BETA;

    const ALPHA: u32 = sbox::ALPHA;
    const INV_ALPHA: Felt = sbox::INV_ALPHA;
    const BETA: u32 = sbox::BETA;
    const DELTA: Felt = sbox::DELTA;

    fn exp_by_inv_alpha(x: Felt) -> Felt {
        sbox::exp_by_inv_alpha(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                MontFp!(
                    "425612051023049830661331964797630809317609247582738116482531241269318882261"
                ),
                MontFp!(
                    "6286665573090460997984811423739073027395747166976635104983563104247751907610"
                ),
            ],
            [
                MontFp!(
                    "19327439051817945875929601061940468337695675737570602169956858066461305609310"
                ),
                MontFp!(
                    "15523588279034506971465434467836619177344673686036121335403340230160460750484"
                ),
            ],
            [
                MontFp!(
                    "13946984741791239153095182933911078852119532427363092061276846287464962360645"
                ),
                MontFp!(
                    "2383989537493311956238507314947336152945718391675018155154321640081436200691"
                ),
            ],
            [
                MontFp!(
                    "6991903415219559939024930622291105204618315276116747230203502695370981043310"
                ),
                MontFp!(
                    "5747075857908422838645819603356939772276027837003424855983408551106681368847"
                ),
            ],
            [
                MontFp!(
                    "7562173524996982196596758348209930832672668661835298571718169643433991656993"
                ),
                MontFp!(
                    "6686830087270821714633181832288499234507418022983405683837375098336060634477"
                ),
            ],
            [
                MontFp!(
                    "20428212542790531737713756862720387151689495909348145408377016074224455016043"
                ),
                MontFp!(
                    "481894954452086852360882110335298590527803701288070320703959607437726776441"
                ),
            ],
        ];

        let output = [
            [
                MontFp!(
                    "14592161914559516814830937163504850059130874104865215775126025263096817472389"
                ),
                Felt::zero(),
            ],
            [
                MontFp!(
                    "17996719371234841169161410606729290168301197702344199238128514626101631218585"
                ),
                MontFp!(
                    "21545190135371267395740182509034156671351839434575279208720937555061390978409"
                ),
            ],
            [
                MontFp!(
                    "1930371117197560939648169063786210033268527574553565215067823407828254316129"
                ),
                MontFp!(
                    "16568070555881000392443170152882554620064457026586610130358418531353270763752"
                ),
            ],
            [
                MontFp!(
                    "14592161914559516814830937163504850059130874104865215775126025263096817472393"
                ),
                MontFp!(
                    "21888242871839275222246405745257275088696311157297823662689037894645226208582"
                ),
            ],
            [
                MontFp!(
                    "3554826803975766712507521152444130750651880154916522114961447795329439631749"
                ),
                MontFp!(
                    "16701744296679158903407912573530508198806112523671685612662923678839880989589"
                ),
            ],
            [
                MontFp!(
                    "5424050821122697031332401789569380486259809291641199341536398137665651914808"
                ),
                MontFp!(
                    "13079723857120649183288605325610774139645530075839366812588796385929570659040"
                ),
            ],
            [
                MontFp!(
                    "19127227924394679064467378445258826196286315962308314101154216806043972337830"
                ),
                MontFp!(
                    "7884688865625967182625059291102471659578410127091859572743776755621420509101"
                ),
            ],
            [
                MontFp!(
                    "18686703634372511939091880127274007619438408722377262904488368324196973279220"
                ),
                MontFp!(
                    "13410331858351132252227237214077307876056542870818045993383924666744077712471"
                ),
            ],
            [
                MontFp!(
                    "20196513515301612096504097013709050879824982768862222092597598984577407390469"
                ),
                MontFp!(
                    "21778956401919562853629280279014195815765214970517856408826355478825519252993"
                ),
            ],
            [
                MontFp!(
                    "11595197418546543019948994980428961007592761116159739713336420913123757597839"
                ),
                MontFp!(
                    "12576256282425290017500099155218513063770784879167690421647901213178298846699"
                ),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiBn254_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
