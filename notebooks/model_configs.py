A1_CONFIG = {
    "encoder": [
        {"type":"conv","out_channels":16,"kernel_size":6,"stride":1,"padding":2},
        {"type":"conv","out_channels":16,"kernel_size":5,"stride":2,"padding":2},
        {"type":"conv","out_channels":16,"kernel_size":4,"stride":2,"padding":1},
        {"type":"conv","out_channels":16,"kernel_size":3,"stride":2,"padding":1},
        {"type":"conv","out_channels":16,"kernel_size":2,"stride":2,"padding":0},
    ],
    "decoder": [
        {"type":"deconv","out_channels":16,"kernel_size":2,"stride":2,"padding":0,"output_padding":0},
        {"type":"deconv","out_channels":16,"kernel_size":3,"stride":2,"padding":1,"output_padding":1},
        {"type":"deconv","out_channels":16,"kernel_size":4,"stride":2,"padding":1,"output_padding":1},
        {"type":"deconv","out_channels":16,"kernel_size":5,"stride":2,"padding":2,"output_padding":1},
        {"type":"deconv","out_channels":1,"kernel_size":6,"stride":1,"padding":2,"output_padding":0},
    ]
}

A2_CONFIG = {
    "encoder": [
        {"type":"conv","out_channels":16,"kernel_size":6,"stride":1,"padding":2},
        {"type":"conv","out_channels":16,"kernel_size":5,"stride":2,"padding":2},
        {"type":"conv","out_channels":16,"kernel_size":4,"stride":2,"padding":1},
        {"type":"conv","out_channels":16,"kernel_size":3,"stride":2,"padding":1},
        {"type":"conv","out_channels":8,"kernel_size":1,"stride":1,"padding":0},
    ],
    "decoder": [
        {"type":"deconv","out_channels":16,"kernel_size":1,"stride":1,"padding":0,"output_padding":0},
        {"type":"deconv","out_channels":16,"kernel_size":3,"stride":2,"padding":1,"output_padding":1},
        {"type":"deconv","out_channels":16,"kernel_size":4,"stride":2,"padding":1,"output_padding":1},
        {"type":"deconv","out_channels":16,"kernel_size":5,"stride":2,"padding":2,"output_padding":1},
        {"type":"deconv","out_channels":1,"kernel_size":6,"stride":1,"padding":2,"output_padding":0},
    ]
}

A3_CONFIG = {
    "encoder": [
        {"type":"conv","out_channels":16,"kernel_size":6,"stride":1,"padding":2},
        {"type":"conv","out_channels":16,"kernel_size":5,"stride":2,"padding":2},
        {"type":"conv","out_channels":16,"kernel_size":4,"stride":2,"padding":1},
        {"type":"conv","out_channels":16,"kernel_size":3,"stride":2,"padding":1},
        {"type":"conv","out_channels":16,"kernel_size":2,"stride":2,"padding":0},
        {"type":"conv","out_channels":16,"kernel_size":1,"stride":2,"padding":0},
    ],
    "decoder": [
        {"type":"deconv","out_channels":16,"kernel_size":2,"stride":2,"padding":0,"output_padding":0},
        {"type":"deconv","out_channels":16,"kernel_size":3,"stride":2,"padding":1,"output_padding":1},
        {"type":"deconv","out_channels":16,"kernel_size":4,"stride":2,"padding":1,"output_padding":1},
        {"type":"deconv","out_channels":16,"kernel_size":5,"stride":2,"padding":2,"output_padding":1},
        {"type":"deconv","out_channels":1,"kernel_size":6,"stride":1,"padding":2,"output_padding":0},
    ]
}