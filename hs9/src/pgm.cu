#include <stdio.h>
#include <ctype.h>

#include "pgm.h"

/* ------------ */
/* -- PGM IO -- */
/* ------------ */

char *readitem(FILE *file, char *buffer);

/* ------------------------------------------ */
char *readitem(FILE *file, char *buffer)
/* ------------------------------------------ */
// read a word
// public domain function: author is unknown
{
    char *aux;
    int k;

    k=0;
    aux=buffer;
    while (!feof(file))
    {
        *aux=fgetc(file);
        switch(k)
        {
        case 0:
            if (*aux=='#') k=1;
            if (isalnum(*aux)) k=2, aux++;
            break;
        case 1:
            if (*aux==0xA) k=0;
            break;
        case 2:
            if (!isalnum(*aux))
            {
                *aux=0;
                return buffer;
            }
            aux++;
            break;
        }
    }
    *aux=0;
    return buffer;
}

/* --------------------------------------------------------------------------------------- */
unsigned char * LoadPGM_ui8matrix(const char *filename, int *h, int *w)
/* --------------------------------------------------------------------------------------- */
{
    // only for P5 binary type, not for text type

    int height, width, grey;
    unsigned char *m;
    FILE *file;

    char *buffer;

    buffer = (char*) calloc(80, sizeof(char));

    // open file
    file = fopen(filename,"rb");
    if (file==NULL) {
      return NULL;
    }

    // read PGM header
    readitem(file, buffer);
    /*fscanf(fichier, "%s", buffer);*/
    if(strcmp(buffer, "P5") != 0)
      return NULL;

    width  = atoi(readitem(file, buffer));
    height = atoi(readitem(file, buffer));
    grey   = atoi(readitem(file, buffer));
    if(grey != 255)
          return NULL;

    *h = height;
    *w = width;
    m = (unsigned char *)malloc(height*width*sizeof(unsigned char));

    size_t fread_num;
    fread_num = fread(m,sizeof(unsigned char), height*width, file);
    if(fread_num != height*width)
      return NULL;

    fclose(file);
    free(buffer);

    return m;
}

/* --------------------------------------------------------------------------------------------- */
int SavePGM_ui8matrix(unsigned char *m, int h, int w, const char *filename)
/* --------------------------------------------------------------------------------------------- */
{
    char buffer[80];

    FILE *file;

    file = fopen(filename, "wb");
    if (file == NULL)
      return 1;

    /* enregistrement de l'image au format pgm */
    sprintf(buffer,"P5\n%d %d\n255\n",w, h);
    fwrite(buffer,strlen(buffer),1,file);

    fwrite(m, sizeof(unsigned char), w*h, file);

    /* fermeture du fichier */
    fclose(file);
    return 0;
}