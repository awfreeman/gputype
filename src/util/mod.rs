use std::{isize, mem::transmute, ops::Index};

use ttf_parser::{Face, GlyphId, Tag};

pub struct CircularSlice<'a, T>  where T: Copy{
    slice: &'a [T]
}
impl<'a, T> CircularSlice<'a, T> where T: Copy{
    pub fn new(slice: &'a[T]) -> Self {
        if slice.len() > isize::MAX as usize {
            panic!("slice too big!")
        }
        CircularSlice {
            slice
        }
    }
    pub fn len(&self) -> isize {
        self.slice.len() as isize
    }
}
impl<T> Index<isize> for CircularSlice<'_, T> where T: Copy{
    type Output = T;

    fn index(&self, index: isize) -> &Self::Output {
        let modulo = index.abs() as usize % self.slice.len();
        if index < 0 && modulo != 0{
            &self.slice[self.slice.len() - modulo]
        } else { 
            &self.slice[modulo]
        }
    }
}
impl<T> Index<usize> for CircularSlice<'_, T> where T: Copy{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.slice[index % self.slice.len()]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub on_curve: bool,
    pub number: usize,
    pub x: f64,
    pub y: f64
}


pub struct GlyphData {
    pub id: GlyphId,
    pub points: Vec<Point>,
    pub contour_end_pts: Vec<usize>,
    pub min: (f64, f64),
    pub max: (f64, f64)
}

pub struct Flags(u8);

impl Flags{
    pub fn on_curve(&self) -> bool {
        self.0 & 0x01 != 0
    }
    pub fn x_short_vec(&self) -> bool {
        self.0 & 0x02 != 0
    }
    pub fn y_short_vec(&self) -> bool {
        self.0 & 0x04 != 0
    }
    pub fn repeat(&self) -> bool {
        self.0 & 0x08 != 0
    }
    pub fn x_is_same_or_positive_short(&self) -> bool {
        self.0 & 0x10 != 0 
    }
    pub fn y_is_same_or_positive_short(&self) -> bool {
        self.0 & 0x20 != 0
    }
}

pub fn inv_lerp(lower: f32, upper: f32, v: f32) -> f32 
{
    ((v - lower) / (upper - lower)).clamp(0., 1.)
}

pub fn read_int16(d: &[u8]) -> i16 {
    unsafe{transmute(read_uint16(d))}
}
pub fn read_uint16(d: &[u8]) -> u16 {
    let d: [u8;2] = [d[1], d[0]];
    u16::from_le_bytes(d)
}

fn get_glyph_bounds(glyph: GlyphId, font: &Face<'_>) -> (usize, usize) {
    let face = font.raw_face();
    let loca = face.table(Tag::from_bytes(b"loca")).unwrap();
    let mut index = glyph.0 as usize;
    let offset: usize;
    let end: usize;
    match font.tables().head.index_to_location_format {
        ttf_parser::head::IndexToLocationFormat::Short => {
            index *= 2;
            offset = read_uint16(&loca[index..]) as usize * 2;
            end = read_uint16(&loca[index+2..]) as usize * 2;
        },
        ttf_parser::head::IndexToLocationFormat::Long => {
            index *= 4;
            offset = u32::from_be_bytes(loca[index..index+4].try_into().unwrap()) as usize;
            end = u32::from_be_bytes(loca[index+4..index+8].try_into().unwrap()) as usize;
        }
    }
    (offset, end)
    
}

pub fn get_glyph_entry(glyph: GlyphId, font: &Face<'_>) -> Option<GlyphData> {
    let (start, end) = get_glyph_bounds(glyph, font);
    if (end - start) == 0 {
        return None
    }
    let glyf = font.raw_face().table(Tag::from_bytes(b"glyf")).unwrap();
    let mut entry = &glyf[start..end];
    let num_contours = read_int16(&entry[0..2]);
    let x_min = read_int16(&entry[2..]);
    let y_min = read_int16(&entry[4..]);
    let x_max = read_int16(&entry[6..]);
    let y_max = read_int16(&entry[8..]);
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
        let _instructions = &entry[..instructions_len];
        entry = &entry[instructions_len..];
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
        let mut accum = 0.;
        for i in 0..num_pts{
            let mut point = Point { on_curve: false, number: i, x: 0., y: 0. };
            if flags[i].x_short_vec() {
                if flags[i].x_is_same_or_positive_short() {
                    accum += entry[ptr] as f64
                } else {
                    accum -= entry[ptr] as f64
                }
                ptr += 1
            } else if !flags[i].x_is_same_or_positive_short() {
                accum += read_int16(&entry[ptr..ptr+2]) as f64;
                ptr += 2;
            }
            point.x = accum;
            point.on_curve = flags[i].on_curve();
            points.push(point);
        }
        //ycoords
        accum = 0.;
        for i in 0..num_pts{
            if flags[i].y_short_vec() {
                if flags[i].y_is_same_or_positive_short() {
                    accum += entry[ptr] as f64
                } else {
                    accum -= entry[ptr] as f64
                }
                ptr += 1
            } else if !flags[i].y_is_same_or_positive_short() {
                accum += read_int16(&entry[ptr..ptr+2]) as f64;
                ptr += 2;
            }
            points[i].y=accum;
        }
        Some(GlyphData {
            id: glyph,
            points,
            contour_end_pts: end_pts_of_contours,
            min: (x_min.into(), y_min.into()),
            max: (x_max.into(), y_max.into())
        })
    } else {
        //compound
        let mut ptr = 0;
        let entry = &entry[10..];
        let mut points = Vec::new();
        let mut contour_end_pts = Vec::new();
        loop {
            let component_flags = read_uint16(&entry);
            assert!(component_flags & 0x0800 == 0);
            let glyph_index = read_uint16(&entry[ptr+2..]);
            ptr += 4;
            let args_are_words = component_flags & 0x8000 != 0;
            let args_are_xy_vals = component_flags & 0x4000 != 0;
            let round_xy_to_grid = component_flags & 0x2000 != 0;
            let we_have_a_scale = component_flags & 0x1000 != 0;
            let more_to_follow = component_flags & 0x0400 != 0;
            let xy_scale_ne = component_flags & 0x0200 != 0;
            let twobytwo_xform = component_flags & 0x0100 != 0;
            let we_have_instr = component_flags & 0x0080 != 0;
            let overlap_compound = component_flags & 0x0040 != 0;
            let (e, f) = {
                if args_are_words {
                    ptr += 4;
                    (read_int16(&entry[ptr-4..]) as f64, read_int16(&entry[ptr-2..]) as f64)
                } else {
                    ptr += 2;
                    unsafe {(transmute::<u8, i8>(entry[ptr-2]) as f64, transmute::<u8, i8>(entry[ptr-1]) as f64)}
                }
            };

            assert!(args_are_xy_vals);
            let (a, b, c, d) = {
                let first = read_int16(&entry[ptr..])as f64 / 16384.0;
                let second = read_int16(&entry[ptr+2..]) as f64 / 16384.0;
                let third = read_int16(&entry[ptr+4..]) as f64 / 16384.0;
                let fourth = read_int16(&entry[ptr+6..]) as f64 / 16384.0;
                if we_have_a_scale {
                    ptr += 2;
                    (first, 0., 0., first)
                } else if xy_scale_ne {
                    ptr += 4;
                    (first, 0., 0., second)
                } else if twobytwo_xform {
                    ptr += 8;
                    (first, second, third, fourth)
                } else {
                    (1., 0., 0., 1.)
                }
            };
            let mut component_info = get_glyph_entry(GlyphId(glyph_index), font);
            if let Some(mut component) = component_info{
                component.points.iter_mut().for_each(|pt: &mut Point| {
                    let x = pt.x;
                    let y = pt.y;
                    pt.x = x * a + y * c + e;
                    pt.y = x * b + y * d + f;
                });
                let offset = points.len();
                points.append(&mut component.points);
                component.contour_end_pts.iter_mut().for_each(|e| *e += offset);
                contour_end_pts.append(&mut component.contour_end_pts);
            }
            if !more_to_follow {
                break
            }

        }
        let (min, max) = {
            let (mut lx, mut ly, mut hx, mut hy) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
            for p in &points {
                if p.x < lx {
                    lx = p.x;
                }
                if p.y < ly {
                    ly = p.y;
                }
                if p.x > hx {
                    hx = p.x
                }
                if p.y > hy {
                    hy = p.y
                }
            }
            ((lx, ly), (hx, hy))
            
        };
        Some(GlyphData {
            id: glyph,
            points,
            contour_end_pts,
            min,
            max

        })
    }
}