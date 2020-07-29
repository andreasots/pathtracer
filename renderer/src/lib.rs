use anyhow::{Context, Error};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

impl Vertex {
    fn rectangle(half_width: f32, half_height: f32) -> [Vertex; 4] {
        [
            Vertex {
                pos: [-half_width, -half_height],
                uv: [0.0, 1.0],
            },
            Vertex {
                pos: [-half_width, half_height],
                uv: [0.0, 0.0],
            },
            Vertex {
                pos: [half_width, -half_height],
                uv: [1.0, 1.0],
            },
            Vertex {
                pos: [half_width, half_height],
                uv: [1.0, 0.0],
            },
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
    let surface = wgpu::Surface::create(&window);
    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
        },
        wgpu::BackendBit::all(),
    )
    .await
    .context("failed to request an adapter")?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await;

    let texture_descriptor = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
    };

    let texture = device.create_texture(&texture_descriptor);
    let texture_view = texture.create_default_view();

    let frag_shader =
        device.create_shader_module(&include!(concat!(env!("OUT_DIR"), "/shader.frag.spv.rs")));
    let vert_shader =
        device.create_shader_module(&include!(concat!(env!("OUT_DIR"), "/shader.vert.spv.rs")));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::FRAGMENT,
            ty: wgpu::BindingType::SampledTexture {
                dimension: wgpu::TextureViewDimension::D2,
                component_type: wgpu::TextureComponentType::Float,
                multisampled: false,
            },
        }],
        label: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        bindings: &[wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&texture_view),
        }],
        layout: &bind_group_layout,
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor {
            module: &vert_shader,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
            module: &frag_shader,
            entry_point: "main",
        }),
        rasterization_state: None,
        primitive_topology: wgpu::PrimitiveTopology::TriangleStrip,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        vertex_state: wgpu::VertexStateDescriptor {
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[wgpu::VertexBufferDescriptor {
                stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float2],
            }],
        },
        sample_count: 1,
        sample_mask: u32::MAX,
        alpha_to_coverage_enabled: false,
    });

    let mut swap_chain_descriptor = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: width as u32,
        height: width as u32,
        present_mode: wgpu::PresentMode::Mailbox,
    };

    let mut swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

    let mut is_done = done.load(Ordering::Acquire);

    let mut vertex_buffer = device.create_buffer_with_data(
        bytemuck::cast_slice(&Vertex::rectangle(1.0, 1.0)),
        wgpu::BufferUsage::VERTEX,
    );

    let mut framebuffer = vec![0.0; 4 * width * height];

    event_loop.run(move |event, _event_loop, control_flow| match event {
        Event::NewEvents(..) => {
            *control_flow = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(100));
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            swap_chain_descriptor.width = size.width;
            swap_chain_descriptor.height = size.height;
            swap_chain = device.create_swap_chain(&surface, &swap_chain_descriptor);

            let scale_w = size.width as f32 / width as f32;
            let scale_h = size.height as f32 / height as f32;

            let vertices = if scale_w < scale_h {
                Vertex::rectangle(1.0, scale_w / scale_h)
            } else {
                Vertex::rectangle(scale_h / scale_w, 1.0)
            };

            vertex_buffer = device.create_buffer_with_data(
                bytemuck::cast_slice(&vertices),
                wgpu::BufferUsage::VERTEX,
            );
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Released,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                },
            ..
        } => match key {
            VirtualKeyCode::Key0 | VirtualKeyCode::Numpad0 => {
                window.set_inner_size(winit::dpi::PhysicalSize::new(width as u32, height as u32))
            }
            _ => (),
        },
        Event::MainEventsCleared => {
            if !is_done {
                window.request_redraw();
            }
            is_done = done.load(Ordering::Acquire);
        }
        Event::RedrawRequested(..) => {
            let frame = swap_chain
                .get_next_texture()
                .expect("failed to get the next frame in the swap chain");
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            for (dst, src) in framebuffer.iter_mut().zip(buffer.iter()) {
                *dst = f32::from_bits(src.load(Ordering::Relaxed));
            }

            let framebuffer_buffer = device.create_buffer_with_data(
                bytemuck::cast_slice(&framebuffer),
                wgpu::BufferUsage::COPY_SRC,
            );

            encoder.copy_buffer_to_texture(
                wgpu::BufferCopyView {
                    buffer: &framebuffer_buffer,
                    offset: 0,
                    bytes_per_row: (4 * width * std::mem::size_of::<f32>()) as u32,
                    rows_per_image: 0,
                },
                wgpu::TextureCopyView {
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    texture: &texture,
                    array_layer: 0,
                },
                wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth: 1,
                },
            );

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: wgpu::Color::BLACK,
                    }],
                    depth_stencil_attachment: None,
                });
                pass.set_pipeline(&render_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_vertex_buffer(
                    0,
                    &vertex_buffer,
                    0,
                    4 * std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                );
                pass.draw(0..4, 0..1);
            }

            queue.submit(&[encoder.finish()]);
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        _ => (),
    })
}