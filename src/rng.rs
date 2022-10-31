use std::cell::UnsafeCell;
use std::rc::Rc;

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

thread_local! {
    static THREAD_PRNG: Rc<UnsafeCell<Xoshiro256PlusPlus>> = {
        Rc::new(UnsafeCell::new(Xoshiro256PlusPlus::from_entropy()))
    }
}

/// A reference to a thread-local pseudorandom number generator.
///
/// Unlike `rand::thread_rng()` this generator is not cryptographically secure.
pub struct ThreadPrng {
    rng: Rc<UnsafeCell<Xoshiro256PlusPlus>>,
}

impl ThreadPrng {
    pub fn get() -> Self {
        Self { rng: THREAD_PRNG.with(|r| r.clone()) }
    }
}

// SAFETY: `self.rng` is only accessed from a single thread and the implementation of `RngCore` for
//  the underlying generator cannot use `ThreadPrng`. Also `rand` is doing the same thing.
impl RngCore for ThreadPrng {
    fn next_u32(&mut self) -> u32 {
        unsafe { &mut *self.rng.get() }.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        unsafe { &mut *self.rng.get() }.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        unsafe { &mut *self.rng.get() }.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        unsafe { &mut *self.rng.get() }.try_fill_bytes(dest)
    }
}
