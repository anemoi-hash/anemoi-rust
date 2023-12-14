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

/// The number of rounds is set to 19 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 19;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over BLS_12_377 scalarfield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiEdOnBls12_377_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiEdOnBls12_377_2_1 {
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
                    "1702453687599237255979790395649661561097021816012655343362862401487976990635"
                ),
                MontFp!(
                    "55182996011466939256354811689314399677205058012274460919631500147033422257"
                ),
            ],
            [
                MontFp!(
                    "1632494083105067989108986168333267989585126749626213300549214124432057272046"
                ),
                MontFp!(
                    "3235496430022085959711123699858140529588122538980640806799684802463067259514"
                ),
            ],
            [
                MontFp!(
                    "2949107286382242589408502806237544591060658194210167305879560683706384630054"
                ),
                MontFp!(
                    "5737317058995992721813276965883365157482803873025273531093951180764314104104"
                ),
            ],
            [
                MontFp!(
                    "3847486564537860245300248276730767485325621899141671582468959814421470544337"
                ),
                MontFp!(
                    "973237804068485700945652982695350747348341011305922079929829613120301223311"
                ),
            ],
            [
                MontFp!(
                    "4423853261878949329714672248313364658249442583402321938245105532670762310457"
                ),
                MontFp!(
                    "6318194943468624511407133555855689372627131855220925784755298465943973848075"
                ),
            ],
            [
                MontFp!(
                    "4016933957638346207048092395830140608782114470984531468566517135441433765388"
                ),
                MontFp!(
                    "6580290746099031704776305709158249319498017489547605549857374926087757457095"
                ),
            ],
        ];

        let output = [
            [
                MontFp!(
                    "1151517511285686876033930673470210890642168091157372340172986380352373987142"
                ),
                Felt::zero(),
            ],
            [
                MontFp!(
                    "6365899358248349268142114350217140688230280935934585545650923395666961413055"
                ),
                MontFp!(
                    "6687234439711495246728724424832755914754252582501095871411974444826560185425"
                ),
            ],
            [
                MontFp!(
                    "1259102676294307221997675460062050531223883403969519571426325121918383273022"
                ),
                MontFp!(
                    "7394748313577717922329439561482524665112163885958294194408694881596009888759"
                ),
            ],
            [
                MontFp!(
                    "1151517511285686876033930673470210890642168091157372340172986380352373987165"
                ),
                MontFp!(
                    "8444461749428370424248824938781546531375899335154063827935233455917409239040"
                ),
            ],
            [
                MontFp!(
                    "535529169994471980895146329785844261993421713552949774502675597694055133100"
                ),
                MontFp!(
                    "6852110193643998530572290501436322032364131063709360756262737129386670745800"
                ),
            ],
            [
                MontFp!(
                    "6404503483962152111999193593477106169718903963947857677408991893720452801303"
                ),
                MontFp!(
                    "3545638184610686040724127622788202781708854453867541694780205848502020435521"
                ),
            ],
            [
                MontFp!(
                    "4993699593849262606727899948154774092954206406493293170903337588327603405059"
                ),
                MontFp!(
                    "538424993017044564736852938526516857998028484060590180933928732894064637201"
                ),
            ],
            [
                MontFp!(
                    "373173095181179190656605172080887661927786715414912434674879791384681579881"
                ),
                MontFp!(
                    "2638495719239556240909523895823467781619734744768035188467450069547637095177"
                ),
            ],
            [
                MontFp!(
                    "4536971513421002575004449422638957461701416375809831818917785571183330345849"
                ),
                MontFp!(
                    "1568913694550148515182306533723005891668126117649843193805645866218823885313"
                ),
            ],
            [
                MontFp!(
                    "5701434605102992787563486029841924857055025452256972247896818057472115916834"
                ),
                MontFp!(
                    "1818001660213901240413679847325689397505131467152159675046705165477840388615"
                ),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiEdOnBls12_377_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
