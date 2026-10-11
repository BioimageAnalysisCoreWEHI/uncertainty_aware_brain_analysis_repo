for (z = 478; z <=536; z++) {
    setSlice(z);
    getSelectionBounds(x0, y0, w, h);

    for (y = y0; y < y0+h; y++) {
        for (x = x0; x < x0+w; x++) {
            if (selectionContains(x, y)) {
                setPixel(x, y, 0);
            }
        }
    }
}