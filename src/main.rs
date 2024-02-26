use std::{fs::File, io::Read, mem::{self, transmute}, path::Path};
use ttf_parser::{Face, GlyphId, Tag};


use std::{error::Error, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage}, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    }, descriptor_set::{allocator::StandardDescriptorSetAllocator, layout::DescriptorBindingFlags, PersistentDescriptorSet, WriteDescriptorSet}, device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags
    }, format::{self, Format}, image::{sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo}, view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        }, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
    }, render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass}, swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    }, sync::{self, GpuFuture}, DeviceSize, Validated, VulkanError, VulkanLibrary
};
use winit::{
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    raw_window_handle::HasDisplayHandle,
    window::WindowBuilder,
};


#[derive(Debug, Clone)]
struct Point {
    on_curve: bool,
    x: isize,
    y: isize
}

struct GlyphData {
    points: Vec<Point>,
    contour_end_pts: Vec<usize>,
    min: (isize, isize),
    max: (isize, isize)
}

struct Flags(u8);

impl Flags{
    fn on_curve(&self) -> bool {
        self.0 & 0x01 != 0
    }
    fn x_short_vec(&self) -> bool {
        self.0 & 0x02 != 0
    }
    fn y_short_vec(&self) -> bool {
        self.0 & 0x04 != 0
    }
    fn repeat(&self) -> bool {
        self.0 & 0x08 != 0
    }
    fn x_is_same_or_positive_short(&self) -> bool {
        self.0 & 0x10 != 0 
    }
    fn y_is_same_or_positive_short(&self) -> bool {
        self.0 & 0x20 != 0
    }
}

fn inv_lerp(lower: f32, upper: f32, v: f32) -> f32 
{
    ((v - lower) / (upper - lower)).clamp(0., 1.)
}

fn read_int16(d: &[u8]) -> i16 {
    unsafe{transmute(read_uint16(d))}
}
fn read_uint16(d: &[u8]) -> u16 {
    let d: [u8;2] = [d[1], d[0]];
    u16::from_le_bytes(d)
}

fn get_glyph_entry(glyph: GlyphId, font: Face<'_>) -> GlyphData {
    println!("Units per em: {}",font.tables().head.units_per_em);
    let face = font.raw_face();
    let loca = face.table(Tag::from_bytes(b"loca")).unwrap();
    //let loca = font.raw_face().table(Tag::from_bytes(b"loca")).unwrap();
    let mut size: usize;
    match font.tables().head.index_to_location_format {
        ttf_parser::head::IndexToLocationFormat::Short => size=2,
        ttf_parser::head::IndexToLocationFormat::Long => size=4,
    }

    let glyf = face.table(Tag::from_bytes(b"glyf")).unwrap();
    println!("size:{}", size);
    let index = (glyph.0 as usize*size);
    let gindex: usize; 
    let glen: usize;
    match size {
        2 => {
            gindex = read_uint16(&loca[index..]).into();
            glen = read_uint16(&loca[index+2..]) as usize - gindex;

        }
        4 => {
            gindex = u32::from_be_bytes(loca[index..index+4].try_into().unwrap()) as usize;
            glen = u32::from_be_bytes(loca[index+4..index+8].try_into().unwrap()) as usize - gindex;
        },
        e => panic!("invalid length: {}", e)
    };

    let mut entry = &glyf[gindex..gindex+glen];
    let num_contours = read_int16(&entry[0..2]);
    let x_min = read_int16(&entry[2..4]);
    let y_min = read_int16(&entry[4..6]);
    let x_max = read_int16(&entry[6..8]);
    let y_max = read_int16(&entry[8..10]);
    if num_contours >= 0 {
        entry = &entry[10..];
        let num_contours = num_contours as usize;
        let end_pts_of_contours = {
            let endpts = entry[0..num_contours*2].chunks_exact(2);
            let mut res = Vec::with_capacity(num_contours/2);
            for pt in endpts {
                res.push(read_uint16(pt) as usize);
            }
            entry = &entry[num_contours*2..];
            res
            
        };
        let num_pts = *end_pts_of_contours.last().unwrap() as usize +1;
        let instructions_len = read_uint16(&entry[0..2]) as usize;
        entry = &entry[2..];
        let instructions = &entry[..instructions_len];
        entry = &entry[instructions_len..];
        print!("Contours: {num_contours}\nCoordinate Range: {x_min},{y_min} to {x_max},{y_max}\nNum. points:{num_pts}\nNum. Instructions:{instructions_len}\n");
        let mut ptr = 0;
        let mut logical_pt = 0;
        let mut flag = Flags(0);
        let mut flags = Vec::new();
        let mut repeat = 0;
        while logical_pt < num_pts {
            if repeat > 0 {
                repeat-=1;
            } else {
                flag = Flags(entry[ptr]);
                if flag.repeat() {
                    ptr += 1;
                    repeat = entry[ptr]
                }
                ptr += 1;
            }
            flags.push(Flags(flag.0));
            logical_pt += 1;
        }
        let mut points = Vec::with_capacity(num_pts);
        //xcoords
        let mut accum = 0;
        for i in 0..num_pts{
            let mut point = Point { on_curve: false, x: 0, y: 0 };
            if flags[i].x_short_vec() {
                if flags[i].x_is_same_or_positive_short() {
                    accum += entry[ptr] as isize
                } else {
                    accum -= entry[ptr] as isize
                }
                ptr += 1
            } else if !flags[i].x_is_same_or_positive_short() {
                accum += read_int16(&entry[ptr..ptr+2]) as isize;
                ptr += 2;
            }
            point.x = accum;
            point.on_curve = flags[i].on_curve();
            points.push(point);
        }
        //ycoords
        accum = 0;
        for i in 0..num_pts{
            if flags[i].y_short_vec() {
                if flags[i].y_is_same_or_positive_short() {
                    accum += entry[ptr] as isize
                } else {
                    accum -= entry[ptr] as isize
                }
                ptr += 1
            } else if !flags[i].y_is_same_or_positive_short() {
                accum += read_int16(&entry[ptr..ptr+2]) as isize;
                ptr += 2;
            }
            points[i].y=accum;
        }
        println!("Points:{:?}", points);
        GlyphData {
            points,
            contour_end_pts: end_pts_of_contours,
            min: (x_min.into(), y_min.into()),
            max: (x_max.into(), y_max.into())
        }
    } else {
        //compound
        todo!("I don't know how to handle composite glyphs");
    }
}

fn main() {
    let path = Path::new("LiberationMono-Regular.ttf");
    let mut file = File::open(path).unwrap();
    let mut content = Vec::new();
    file.read_to_end(&mut content).unwrap();
    let mut font = Face::parse(&content, 0).unwrap();
    let glyph = font.glyph_index('a').unwrap();
    let points = get_glyph_entry(glyph, font);
    let event_loop = EventLoop::new().unwrap();

    let library = VulkanLibrary::new().unwrap();

    // The first step of any Vulkan program is to create an instance.
    //
    // When we create an instance, we have to pass a list of extensions that we want to enable.
    //
    // All the window-drawing functionalities are part of non-core extensions that we need to
    // enable manually. To do so, we ask `Surface` for the list of extensions required to draw to
    // a window.
    let required_extensions = Surface::required_extensions(&event_loop);

    // Now creating the instance.
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            // Enable enumerating devices that use non-conformant Vulkan implementations.
            // (e.g. MoltenVK)
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    // We then choose which physical device to use. First, we enumerate all the available physical
    // devices, then apply filters to narrow them down to those that can support our needs.
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            // Some devices may not support the extensions or features that your application, or
            // report properties and limits that are not sufficient for your application. These
            // should be filtered out here.
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // For each physical device, we try to find a suitable queue family that will execute
            // our draw commands.
            //
            // Devices can provide multiple queues to run commands in parallel (for example a draw
            // queue and a compute queue), similar to CPU threads. This is something you have to
            // have to manage manually in Vulkan. Queues of the same type belong to the same queue
            // family.
            //
            // Here, we look for a single queue family that is suitable for our purposes. In a
            // real-world application, you may want to use a separate dedicated transfer queue to
            // handle data transfers in parallel with graphics operations. You may also need a
            // separate queue for compute operations, if your application uses those.
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // We select a queue family that supports graphics operations. When drawing to
                    // a window surface, as we do in this example, we also need to check that
                    // queues in this queue family are capable of presenting images to the surface.
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                // The code here searches for the first queue family that is suitable. If none is
                // found, `None` is returned to `filter_map`, which disqualifies this physical
                // device.
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("no suitable physical device found");
    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();
    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-1., -1.],
        },
        Vertex {
            position: [1., -1.],
        },
        Vertex {
            position: [1., 1.],
        },
        Vertex {
            position: [1., 1.],
        },
        Vertex {
            position: [-1., 1.],
        },
        Vertex {
            position: [-1., -1.]
        }
        
    ];
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();
    let (len_buf, curve_buf) = {
        let (lx, ly) = points.min;
        let (hx, hy) = points.max;
        let interp = |x: isize, y: isize| [
            inv_lerp(lx as f32, hx as f32, x as f32),
            inv_lerp(ly as f32, hy as f32, y as f32)
        ];
        let mid = |a:[f32;2], b:[f32;2]| [
            (a[0]+b[0])/2., (a[1]+b[1])/2.
        ];
        let mut curves = Vec::new();
        let mut prev = 0;
        for end in points.contour_end_pts {
            let contour = &points.points[prev..=end];
            prev = end+1;
            let mut i = 0;
            while i < contour.len() {
                let next = &contour[(i+1)%contour.len()];
                let curr = &contour[i];
                let prev = &contour[if i == 0 {contour.len()-1} else {i-1}];
                let mut p0;
                let mut p1;
                let mut p2;
                if !curr.on_curve {

                    p0 = interp(prev.x, prev.y);
                    p1 = interp(curr.x, curr.y);
                    p2 = interp(next.x, next.y);
                    if !prev.on_curve {
                        p0 = mid(p0, p1);
                    } 
                    if !next.on_curve {
                        p2 = mid(p1, p2);
                    }
                    i += 1;
                } else {
                    p0 = interp(curr.x, curr.y);
                    if next.on_curve {
                        p2 = interp(next.x, next.y);
                        p1 = mid(p0, p2);
                        i += 1;
                    } else {
                        let last = &contour[(i+2)%contour.len()];
                        p1 = interp(next.x, next.y);
                        p2 = interp(last.x, last.y);
                        if !last.on_curve{
                            p2=mid(p1, p2);
                        }
                        i += 2;
                    }
                }
                curves.push([p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]]);
                
            }
        }
        for cv in &curves {
            if cv[3] == 1.0 {
                println!("{:?}", cv)
            }
        };
        (
            Buffer::from_data(
                memory_allocator.clone(), 
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                }, 
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                }, curves.len() as u32).unwrap(),
            Buffer::from_iter(
                memory_allocator.clone(), 
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                }, 
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                }, 
                curves).unwrap(),
            )

    };
    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/vertex.glsl"
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/frag.glsl",
        }
    }

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },

        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap();

    let sampler = Sampler::new(device.clone(), SamplerCreateInfo{
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        address_mode: [SamplerAddressMode::Repeat; 3],
        ..Default::default()
    }).unwrap();
    
    let pipeline = {
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = {
            let mut lci = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
            PipelineLayout::new(
                device.clone(),
                lci.into_pipeline_layout_create_info(device.clone()).unwrap(),
                ).unwrap()
        };
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(), 
                    ColorBlendAttachmentState::default()
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            }
        ).unwrap()

    };
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator, 
        layout.clone(), 
        [
            WriteDescriptorSet::buffer(0, curve_buf),
            WriteDescriptorSet::buffer(1, len_buf)],
        []).unwrap();
    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0
    };

    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let command_buffer_allocator = 
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let _ = event_loop.run(move |event,wintarget| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                wintarget.exit()
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::AboutToWait => {
                // Do not draw the frame when the screen size is zero. On Windows, this can
                // occur when minimizing the application.
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                // It is important to call this function from time to time, otherwise resources
                // will keep accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU
                // has already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the
                // window size. In this example that includes the swapchain, the framebuffers and
                // the dynamic state viewport.
                if recreate_swapchain {
                    // Use the new dimensions of the window.

                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    swapchain = new_swapchain;

                    // Because framebuffers contains a reference to the old swapchain, we need to
                    // recreate framebuffers as well.
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );

                    recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the
                // swapchain. If no image is available (which happens if you submit draw commands
                // too quickly), then the function will block. This operation returns the index of
                // the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional
                // timeout after which the function call will return an error.
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }

                // In order to draw, we have to build a *command buffer*. The command buffer object
                // holds the list of commands that are going to be executed.
                //
                // Building a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    // Before we can draw, we have to *enter a render pass*.
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            // A list of values to clear the attachments with. This list contains
                            // one item for each attachment in the render pass. In this case, there
                            // is only one attachment, and we clear it with a blue color.
                            //
                            // Only attachments that have `AttachmentLoadOp::Clear` are provided
                            // with clear values, any others should use `None` as the clear value.
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],

                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        SubpassBeginInfo {
                            // The contents of the first (and only) subpass.
                            // This can be either `Inline` or `SecondaryCommandBuffers`.
                            // The latter is a bit more advanced and is not covered here.
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    // We are now inside the first subpass of the render pass.
                    //
                    // TODO: Document state setting and how it affects subsequent draw commands.
                    .set_viewport(0, [viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(), 
                        0, 
                        set.clone())
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    // We add a draw command.
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    // We leave the render pass. Note that if we had multiple subpasses we could
                    // have called `next_subpass` to jump to the next subpass.
                    .end_render_pass(Default::default())
                    .unwrap();

                // Finish building the command buffer by calling `build`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    // The color output is now expected to contain our triangle. But in order to
                    // show it on the screen, we have to *present* the image by calling
                    // `then_swapchain_present`.
                    //
                    // This function does not actually present the image immediately. Instead it
                    // submits a present command at the end of the queue. This means that it will
                    // only be presented once the GPU has finished executing the command buffer
                    // that draws the triangle.
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        panic!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => (),
        }
    });

}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                }
            ).unwrap()
        }).collect::<Vec<_>>()
}