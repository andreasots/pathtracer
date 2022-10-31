use anyhow::{Context, Error};
use std::io::Write;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

impl Vertex {
    fn rectangle(half_width: f32, half_height: f32) -> [Vertex; 4] {
        [
            Vertex { pos: [-half_width, -half_height], uv: [0.0, 1.0] },
            Vertex { pos: [-half_width, half_height], uv: [0.0, 0.0] },
            Vertex { pos: [half_width, -half_height], uv: [1.0, 1.0] },
            Vertex { pos: [half_width, half_height], uv: [1.0, 0.0] },
        ]
    }
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}

pub async fn renderer(
    width: usize,
    height: usize,
    title: &str,
    buffer: Arc<Vec<AtomicU32>>,
    done: Arc<AtomicBool>,
) -> Result<(), Error> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(width as u32, height as u32))
        .with_title(title)
        .build(&event_loop)
        .context("failed to create the window")?;
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    // TODO: safety
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .context("failed to request an adapter")?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .context("failed to request a device")?;

    let texture_descriptor = wgpu::TextureDescriptor {
        label: Some("texture_descriptor"),
        size: wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    };

    let texture = device.create_texture(&texture_descriptor);
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: 0.0,
        lod_max_clamp: 0.0,
        compare: None,
        anisotropy_clamp: None,
        border_color: None,
    });

    let tone_mapping_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tone_mapping_buffer"),
        contents: bytemuck::cast_slice(&[0.18f32, 1.0f32, 0.0f32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &tone_mapping_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(wgpu::IndexFormat::Uint16),
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
            unclipped_depth: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: u64::MAX,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8Unorm,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
    });

    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: width as u32,
        height: width as u32,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
    };
    surface.configure(&device, &surface_config);

    let mut vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertex_buffer"),
        contents: bytemuck::cast_slice(&Vertex::rectangle(1.0, 1.0)),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut framebuffer = vec![0.0; 4 * width * height];
    let mut is_done = done.load(Ordering::Acquire);
    let mut active = true;
    let mut avg_zone = 5.0;
    let mut white_zone = 10.0;

    event_loop.run(move |event, _event_loop, control_flow| match event {
        Event::NewEvents(..) => {
            *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(100));
        }
        Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
            surface_config.width = size.width;
            surface_config.height = size.height;
            surface.configure(&device, &surface_config);

            let scale_w = size.width as f32 / width as f32;
            let scale_h = size.height as f32 / height as f32;

            let vertices = if scale_w < scale_h {
                Vertex::rectangle(1.0, scale_w / scale_h)
            } else {
                Vertex::rectangle(scale_h / scale_w, 1.0)
            };

            vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        }
        Event::WindowEvent { event: WindowEvent::ReceivedCharacter(c), .. } => {
            match c {
                '0' => {
                    window
                        .set_inner_size(winit::dpi::PhysicalSize::new(width as u32, height as u32));
                    avg_zone = 5.0;
                    white_zone = 10.0;
                }
                '+' => avg_zone += 0.25,
                '-' => avg_zone -= 0.25,
                '*' => white_zone += 0.25,
                '/' => white_zone -= 0.25,
                'f' => {
                    let fullscreen = if window.fullscreen().is_none() {
                        Some(Fullscreen::Borderless(window.current_monitor()))
                    } else {
                        None
                    };
                    window.set_fullscreen(fullscreen);
                }
                _ => (),
            }
            window.request_redraw();
        }
        Event::WindowEvent { event: WindowEvent::Focused(focused), .. } => {
            active = focused;
            if active && is_done {
                window.request_redraw()
            }
        }
        Event::MainEventsCleared => {
            if active && !is_done {
                window.request_redraw();
            }
            is_done = done.load(Ordering::Acquire);
        }
        Event::RedrawRequested(..) => {
            let frame = match surface.get_current_texture() {
                Ok(frame) => frame,
                Err(err) => {
                    eprintln!("ERROR: failed to get the next frame in the swap chain: {:?}", err);
                    return;
                }
            };
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let mut log_total_luminance = 0.0;

            for (dst, src) in framebuffer.chunks_mut(4).zip(buffer.chunks(4)) {
                dst[0] = f32::from_bits(src[0].load(Ordering::Relaxed));
                dst[1] = f32::from_bits(src[1].load(Ordering::Relaxed));
                dst[2] = f32::from_bits(src[2].load(Ordering::Relaxed));
                dst[3] = f32::from_bits(src[3].load(Ordering::Relaxed));

                log_total_luminance += (0.001 + dst[1].abs()).ln();
            }
            let avg_luminance = (4.0 * log_total_luminance / framebuffer.len() as f32).exp();
            let max_luminance = avg_luminance * 2.0f32.powf(white_zone - avg_zone);
            let middle_gray = 0.18 * 2.0f32.powf(avg_zone - 5.0);

            {
                let stdout = std::io::stdout();
                let mut stdout = stdout.lock();
                let _ = write!(
                    stdout,
                    "\r\x1B[2Kavg {:8.4} (zone {}) max {:8.4} (zone {})",
                    avg_luminance, avg_zone, max_luminance, white_zone
                );
                let _ = stdout.flush();
            }
            let new_tone_mapping_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("new_tone_mapping_buffer"),
                    contents: bytemuck::cast_slice(&[middle_gray, avg_luminance, max_luminance]),
                    usage: wgpu::BufferUsages::COPY_SRC,
                });
            encoder.copy_buffer_to_buffer(
                &new_tone_mapping_buffer,
                0,
                &tone_mapping_buffer,
                0,
                3 * std::mem::size_of::<f32>() as wgpu::BufferAddress,
            );

            let framebuffer_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("framebuffer_buffer"),
                contents: bytemuck::cast_slice(&framebuffer),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer: &framebuffer_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(
                            NonZeroU32::new((4 * width * std::mem::size_of::<f32>()) as u32)
                                .expect("framebuffer stride is zero"),
                        ),
                        rows_per_image: None,
                    },
                },
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
            );

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });
                pass.set_pipeline(&render_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..4, 0..1);
            }

            queue.submit([encoder.finish()]);
            frame.present();
        }
        Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    })
}
