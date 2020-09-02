#!/bin/bash
# Author: Jan Schlüter
if [ $# -lt 2 ]; then
    echo "Collects and converts audio files to 22 kHz WAV format."
    echo "Runs 8 conversion processes in parallel. Needs ffmpeg."
    echo "Usage: $0 TARGET SOURCE..."
    echo "  TARGET: The target directory, will be created if needed. Can be"
    echo "     just audio/, or a subdirectory."
    echo "  SOURCE: The directory to browse recursively for .wav, .mp3, .ogg,"
    echo "     and .flac files, can be given multiple times. The directory "
    echo "     structure below SOURCE will be recreated in TARGET, but SOURCE "
    echo "     itself will not be included."
fi

target="$1"
i=1
for source in "${@:2}"; do
    while IFS= read -d '' -r infile; do
        outfile="$target/${infile%.*}.wav"
        infile="$source/$infile"
	if [ ! -f "$outfile" ]; then
            outdir="${outfile%/*}"
            mkdir -p "$outdir"
            # display progress on stderr
            >&2 echo -ne "\r\e[K$i: $outfile"  # \r: return, \e[K: delete rest of line
            # write command to stdout (0-terminated)
            echo -ne "ffmpeg -v fatal -nostdin -i \"$infile\" -c:a pcm_s16le -ar 22050 \"$outfile\"\0"
        fi
        ((i++))
    done < <(find -L "$source" \( -name '*.wav' -or -name '*.mp3' \
             -or -name '*.ogg' -or -name '*.flac' \) -printf '%P\0') \
         | xargs --no-run-if-empty -0 -n1 -P8 sh -c
         # execute up to eight commands in parallel
done
>&2 echo
