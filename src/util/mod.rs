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
    pub x: isize,
    pub y: isize
}

pub struct GlyphData {
    pub id: GlyphId,
    pub points: Vec<Point>,
    pub contour_end_pts: Vec<usize>,
    pub min: (isize, isize),
    pub max: (isize, isize)
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

pub fn get_glyph_entry(glyph: GlyphId, font: &Face<'_>) -> GlyphData {
    let face = font.raw_face();
    let loca = face.table(Tag::from_bytes(b"loca")).unwrap();
    let size: usize;
    match font.tables().head.index_to_location_format {
        ttf_parser::head::IndexToLocationFormat::Short => size=2,
        ttf_parser::head::IndexToLocationFormat::Long => size=4,
    }

    let glyf = face.table(Tag::from_bytes(b"glyf")).unwrap();
    let index = glyph.0 as usize*size;
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
        let mut accum = 0;
        for i in 0..num_pts{
            let mut point = Point { on_curve: false, number: i, x: 0, y: 0 };
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
        GlyphData {
            id: glyph,
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