use bvh::nalgebra::{Point3, Vector3};
use bvh::ray::Ray;

pub struct Camera {
    canvas_scale: (f32, f32),
    sensor: (f32, f32),
    focal_length: f32,
}

impl Camera {
    pub fn new(canvas: (f32, f32), sensor_width: f32, focal_length: f32) -> Self {
        Camera {
            canvas_scale: (1.0 / (canvas.0 + 1.0), 1.0 / (canvas.1 + 1.0)),
            sensor: (sensor_width, sensor_width * canvas.1 / canvas.0),
            focal_length,
        }
    }

    pub fn generate_ray(&self, x: f32, y: f32) -> Ray {
        Ray::new(
            Point3::origin(),
            Vector3::new(
                ((x + 0.5) * self.canvas_scale.0 - 0.5) * self.sensor.0,
                // negative because on the screen the Y-axis goes down instead of up.
                -((y + 0.5) * self.canvas_scale.1 - 0.5) * self.sensor.1,
                -self.focal_length,
            ),
        )
    }
}

impl From<crate::scene::Camera> for Camera {
    fn from(camera: crate::scene::Camera) -> Self {
        Self::new(
            (camera.resolution.0 as f32, camera.resolution.1 as f32),
            camera.sensor_width,
            camera.focal_length,
        )
    }
}

#[cfg(test)]
mod test {
    use super::Camera;
    use bvh::nalgebra::Vector3;

    #[test]
    fn corners() {
        let camera = Camera::new((1024.0, 768.0), 36.0, 50.0);

        assert_eq!(
            camera.generate_ray(-0.5, -0.5).direction,
            Vector3::new(-36.0 / 2.0, 27.0 / 2.0, -50.0).normalize()
        );
        assert_eq!(
            camera.generate_ray(1024.5, -0.5).direction,
            Vector3::new(36.0 / 2.0, 27.0 / 2.0, -50.0).normalize()
        );
        assert_eq!(
            camera.generate_ray(-0.5, 768.5).direction,
            Vector3::new(-36.0 / 2.0, -27.0 / 2.0, -50.0).normalize()
        );
        assert_eq!(
            camera.generate_ray(1024.5, 768.5).direction,
            Vector3::new(36.0 / 2.0, -27.0 / 2.0, -50.0).normalize()
        );
    }
}
