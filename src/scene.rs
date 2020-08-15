use crate::bvh::{Ray, BVH};
use crate::material::{Material, D65};
use crate::{hosek_wilkie::HosekWilkieSkyModel, triangle::Triangle};
use anyhow::{Context, Error};
use nalgebra::{Matrix4, Point2, Point3, Vector3, Vector4};
use obj::{IndexTuple, Obj, ObjMaterial};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

trait ObjExt {
    fn triangle(
        &self,
        transform: &Matrix4<f32>,
        a: &IndexTuple,
        b: &IndexTuple,
        c: &IndexTuple,
        material_index: usize,
    ) -> Triangle;
}

impl ObjExt for Obj {
    fn triangle(
        &self,
        transform: &Matrix4<f32>,
        a: &IndexTuple,
        b: &IndexTuple,
        c: &IndexTuple,
        material_index: usize,
    ) -> Triangle {
        let uv = match (a.1, b.1, c.1) {
            (Some(a), Some(b), Some(c)) => Some([
                Point2::from(self.data.texture[a]),
                Point2::from(self.data.texture[b]),
                Point2::from(self.data.texture[c]),
            ]),
            (None, None, None) => None,
            idx => panic!("texture coords defined on some vertices but not all: {:?}", idx),
        };

        let normal = match (a.2, b.2, c.2) {
            (Some(a), Some(b), Some(c)) => Some([
                transform.transform_vector(&Vector3::from(self.data.normal[a])),
                transform.transform_vector(&Vector3::from(self.data.normal[b])),
                transform.transform_vector(&Vector3::from(self.data.normal[c])),
            ]),
            (None, None, None) => None,
            idx => panic!("normals defined on some vertices but not all: {:?}", idx),
        };

        let a = transform.transform_point(&Point3::from(self.data.position[a.0]));
        let b = transform.transform_point(&Point3::from(self.data.position[b.0]));
        let c = transform.transform_point(&Point3::from(self.data.position[c.0]));

        Triangle::new(a, b, c, uv, normal, material_index)
    }
}

#[derive(Deserialize, Serialize, Debug)]
struct SceneFile {
    #[serde(default)]
    camera: Camera,
    #[serde(default)]
    sky: Sky,
    #[serde(default)]
    meshes: Vec<Mesh>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(untagged)]
enum Sky {
    D65 {
        power: f32,
    },
    HosekWilkie {
        atmospheric_turbidity: f32,
        ground_albedo: f32,
        zenith_direction: [f32; 3],
        sun_direction: [f32; 3],
    },
}

impl Default for Sky {
    fn default() -> Self {
        Self::D65 { power: 0.0 }
    }
}

enum InitialisedSky {
    D65 {
        power: f32,
    },
    HosekWilkie {
        model: HosekWilkieSkyModel,
        zenith_direction: Vector3<f32>,
        sun_direction: Vector3<f32>,
    },
}

impl InitialisedSky {
    fn from_sky(sky: Sky, transform: &Matrix4<f32>) -> Self {
        match sky {
            Sky::D65 { power } => Self::D65 { power },
            Sky::HosekWilkie {
                atmospheric_turbidity,
                ground_albedo,
                zenith_direction,
                sun_direction,
            } => {
                let zenith_direction =
                    transform.transform_vector(&Vector3::from(zenith_direction)).normalize();
                let sun_direction =
                    transform.transform_vector(&Vector3::from(sun_direction)).normalize();
                let solar_elevation = zenith_direction.dot(&sun_direction).asin();

                assert!(solar_elevation > 0.0);
                println!("solar elevation: {:7.4}°", solar_elevation.to_degrees());
                let model =
                    HosekWilkieSkyModel::new(solar_elevation, atmospheric_turbidity, ground_albedo);

                Self::HosekWilkie { model, zenith_direction, sun_direction }
            }
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Copy, Clone)]
#[serde(default)]
pub struct Camera {
    pub resolution: (usize, usize),
    pub sensor_width: f32,
    pub focal_length: f32,
    pub samples: u32,
    transform: CameraTransform,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            resolution: (1024, 768),
            sensor_width: 36.0,
            focal_length: 50.0,
            samples: 128,
            transform: CameraTransform::Matrix(Matrix4::identity().into()),
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Copy, Clone)]
#[serde(untagged)]
enum CameraTransform {
    LookAt { eye: [f32; 3], target: [f32; 3], up: [f32; 3] },
    Matrix([[f32; 4]; 4]),
}

impl CameraTransform {
    fn into_matrix(self) -> Matrix4<f32> {
        match self {
            CameraTransform::LookAt { eye, target, up } => {
                Matrix4::look_at_rh(&Point3::from(eye), &Point3::from(target), &Vector3::from(up))
            }
            CameraTransform::Matrix(matrix) => Matrix4::from(matrix),
        }
    }
}

#[derive(Deserialize, Debug, Serialize)]
struct Mesh {
    path: PathBuf,
    #[serde(default = "Mesh::identity_transform")]
    transform: [[f32; 4]; 4],
}

impl Mesh {
    fn identity_transform() -> [[f32; 4]; 4] {
        Matrix4::identity().into()
    }
}

pub struct Scene {
    pub camera: Camera,
    bvh: BVH,
    triangles: Vec<Triangle>,
    materials: Vec<Material>,
    sky: InitialisedSky,
}

impl Scene {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let path = path.as_ref();
        let scene: SceneFile = {
            let mut file = File::open(path).context("failed to open scene file")?;
            let mut contents = vec![];
            file.read_to_end(&mut contents).context("failed to read scene file")?;
            toml::from_slice(&contents).context("failed to parse scene file")?
        };

        let mut triangles = vec![];
        let mut materials = vec![Material::default()];

        let camera_transform = scene.camera.transform.into_matrix();

        for mesh in scene.meshes {
            println!("Loading {:?}", mesh.path);

            let transform = Matrix4::from(mesh.transform) * camera_transform;

            let mesh_path = path.with_file_name(&mesh.path);
            let mut mesh = Obj::load(&mesh_path).context("failed to load the mesh")?;
            mesh.load_mtls().context("failed to load the MTLs")?;

            let mut material_map = HashMap::new();

            for object in &mesh.data.objects {
                for group in &object.groups {
                    let material = match group.material {
                        Some(ObjMaterial::Mtl(ref material)) => match material_map
                            .entry(&material.name)
                        {
                            Entry::Vacant(entry) => {
                                let i = materials.len();
                                let mtl = mesh
                                    .data
                                    .material_libs
                                    .iter()
                                    .find(|mtl| {
                                        mtl.materials.iter().any(|mat| mat.name == material.name)
                                    })
                                    .expect("material was loaded from nowhere???");
                                materials.push(Material::from_mtl(
                                    mesh.path.join(&mtl.filename),
                                    &material,
                                )?);
                                *entry.insert(i)
                            }
                            Entry::Occupied(entry) => *entry.get(),
                        },
                        Some(ObjMaterial::Ref(ref name)) => {
                            eprintln!("missing material {:?}", name);
                            0
                        }
                        None => 0,
                    };

                    for poly in &group.polys {
                        match &poly.0[..] {
                            [] => println!("Ignoring empty polygon"),
                            [a] => println!("Ignoring a lone point: {:?}", a),
                            line @ [_, _] => println!("Ignoring a lone line: {:?}", line),
                            [a, b, c] => {
                                triangles.push(mesh.triangle(&transform, a, b, c, material));
                            }
                            [a, b, c, d] => {
                                triangles.push(mesh.triangle(&transform, a, b, c, material));
                                triangles.push(mesh.triangle(&transform, a, c, d, material));
                            }
                            polygon => {
                                return Err(anyhow::format_err!("unhandled {}-gon", polygon.len()))
                            }
                        }
                    }
                }
            }
        }

        println!("Loaded {} triangles", triangles.len());

        let bvh = BVH::build(&mut triangles);

        Ok(Scene {
            camera: scene.camera,
            bvh,
            triangles,
            materials,
            sky: InitialisedSky::from_sky(scene.sky, &camera_transform),
        })
    }

    fn evaluate_sky(&self, ray: Ray, wavelengths: [f32; 4],) -> Vector4<f32> {
        match self.sky {
            InitialisedSky::D65 { power } if power == 0.0 => Vector4::from_element(0.0),
            InitialisedSky::D65 { power } => D65.sample4(wavelengths) * power,
            InitialisedSky::HosekWilkie { model, zenith_direction, sun_direction } => {
                let cos_theta = ray.direction.dot(&zenith_direction);
                if cos_theta > 0.0 {
                    let theta = cos_theta.min(1.0).acos();
                    let cos_gamma = ray.direction.dot(&sun_direction);
                    let gamma = cos_gamma.min(1.0).max(-1.0).acos();

                    if gamma < model.solar_radius {
                        model.solar_radiance4(theta, cos_theta, gamma, cos_gamma, wavelengths)
                    } else {
                        model.radiance4(cos_theta, gamma, cos_gamma, wavelengths)
                    }
                } else {
                    Vector4::from_element(0.0)
                }
            }
        }
    }

    pub fn radiance<R>(
        &self,
        ray: Ray,
        wavelengths: [f32; 4],
        rng: &mut R,
        start: Option<&Triangle>,
        depth: usize,
    ) -> Vector4<f32>
    where
        R: Rng + ?Sized,
    {
        if depth > 16 {
            return Vector4::from_element(0.0);
        }

        if let Some((triangle, intersection)) = self.bvh.traverse(&ray, start, &self.triangles) {
            self.materials[triangle.material_index()].radiance(
                ray,
                wavelengths,
                intersection,
                triangle,
                self,
                rng,
                depth,
            )
        } else {
            self.evaluate_sky(ray, wavelengths)
        }
    }
}
