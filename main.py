import utils, align, hdr, tonemapping

if __name__ == '__main__':

    images, lnt, alignType, std_img_idx = utils.read_ldr_images('img/test2')
    images = align.align(images, alignType, std_img_idx, 5)
    channels = utils.ldr_to_channels(images)
    hdr_img = hdr.hdr_reconstruction(channels, lnt, 10, False)
    utils.save_hdr_image(hdr_img, 'hdr')
    tonemapping.tonemappingDrago(hdr_img, 'ldr')
    
    print("\nDONE")
