# Gputype: TrueType Font rendering demo
I got aggravated by [SLUG](https://sluglibrary.com/) being closed source so I tried to make my own GPU accelerated text renderer using [Vulkano](https://github.com/vulkano-rs/vulkano). The result is obviously very basic but it does work. 

![Screenshot](./Screenshot.png)
## Libraries you should probably use instead
- [libschrift](https://github.com/tomolt/libschrift) 
- [stb_truetype](https://github.com/nothings/stb/blob/master/stb_truetype.h)
### TODO:
- Grid fitting for instructed fonts
- Actual text layout
- Make it useable as a library
- Supersampling maybe?