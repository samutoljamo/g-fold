// The constraint builders index gravity/component arrays by the same loop
// variable used for the x/u component offsets; an indexed loop is clearer here
// than zipped iterators.
#![allow(clippy::needless_range_loop)]

pub mod config;
pub mod derive;
pub mod assemble;
pub mod solve;
pub mod validate;
