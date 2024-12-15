#pragma once

unsigned char * LoadPGM_ui8matrix(const char *filename, int *h, int *w);

int SavePGM_ui8matrix(unsigned char *m, int h, int w, const char *filename);