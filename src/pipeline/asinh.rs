#[derive(Clone, Copy)]
pub struct AsinhMapping {
    min: f64,
    max: f64,
    beta: f64,
    inv_asinh_range: f64,
}

impl AsinhMapping {
    pub fn from_range(global_min: f64, global_max: f64) -> Self {
        let range = (global_max - global_min).max(f64::EPSILON);
        let beta = 0.02 * range;
        let inv_asinh_range = ((range / beta).asinh()).max(f64::EPSILON).recip();
        Self {
            min: global_min,
            max: global_max,
            beta,
            inv_asinh_range,
        }
    }
    pub fn beta(&self) -> f64 {
        self.beta
    }

    pub fn apply(&self, value: f64) -> f64 {
        let clamped = value.clamp(self.min, self.max);
        let shifted = clamped - self.min;
        ((shifted / self.beta).asinh() * self.inv_asinh_range).clamp(0.0, 1.0)
    }
}
