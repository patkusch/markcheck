import io
import json
import struct

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from markcheck.cli import main
from markcheck.detectors import metadata
from markcheck.filetypes import sniff

IPTC_NS = "http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
DST = "cv.iptc.org/newscodes/digitalsourcetype/"


def run(data: bytes, name="x.bin"):
    from pathlib import Path
    p = Path(name)
    return metadata.check(p, data, sniff(p, data[:64]), {})


def found(findings):
    return [f for f in findings if f.status == "found"]


def xmp(inner="", attrs="", ns=""):
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="{IPTC_NS}" xmlns:xmp="http://ns.adobe.com/xap/1.0/" {ns} {attrs}>'
        f"{inner}</rdf:Description></rdf:RDF></x:xmpmeta>"
    ).encode()


def jpeg(segments=()):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 140, 160)).save(buf, "JPEG")
    raw = buf.getvalue()
    extra = b"".join(b"\xff\xe1" + struct.pack(">H", len(s) + 2) + s for s in segments)
    return raw[:2] + extra + raw[2:]


def jpeg_xmp(packet):
    return jpeg([b"http://ns.adobe.com/xap/1.0/\x00" + packet])


def png_text(**kv):
    info = PngInfo()
    for k, v in kv.items():
        info.add_text(k, v)
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


# ---------------------------------------------------------------- fixtures

def _manifest(fixtures, sub):
    return json.loads((fixtures / sub / "manifest.json").read_text())["files"]


def test_every_metadata_fixture_is_flagged(fixtures):
    entries = _manifest(fixtures, "metadata")
    assert len(entries) >= 7
    for e in entries:
        data = (fixtures / "metadata" / e["file"]).read_bytes()
        fs = run(data, e["file"])
        assert found(fs), (e["file"], [f.summary for f in fs])


def test_controls_give_no_metadata_alarm(fixtures):
    for e in _manifest(fixtures, "controls"):
        data = (fixtures / "controls" / e["file"]).read_bytes()
        fs = run(data, e["file"])
        assert not found(fs), (e["file"], [f.summary for f in fs])
        assert all(f.status == "not_found" for f in fs)


def test_cli_exit_codes(fixtures, capsys):
    assert main([str(fixtures / "metadata")]) == 1
    out = capsys.readouterr().out
    assert "metadata says it was made by AI" in out
    assert "not signed" in out


# ---------------------------------------------------------------- IPTC forms

@pytest.mark.parametrize("value", [
    f"http://{DST}trainedAlgorithmicMedia",
    f"https://{DST}trainedAlgorithmicMedia",
    f"https://{DST}trainedAlgorithmicMedia/",
    "trainedAlgorithmicMedia",
    "digsrctype:trainedAlgorithmicMedia",
])
def test_iptc_value_forms(value):
    fs = found(run(jpeg_xmp(xmp(attrs=f'Iptc4xmpExt:DigitalSourceType="{value}"')), "a.jpg"))
    assert len(fs) == 1
    assert "made by AI" in fs[0].summary and "trainedAlgorithmicMedia" in fs[0].summary


def test_iptc_element_and_resource_forms():
    el = xmp("<Iptc4xmpExt:DigitalSourceType>compositeWithTrainedAlgorithmicMedia</Iptc4xmpExt:DigitalSourceType>")
    res = xmp(f'<Iptc4xmpExt:DigitalSourceType rdf:resource="http://{DST}compositeSynthetic"/>')
    for pkt, code in ((el, "compositeWithTrainedAlgorithmicMedia"), (res, "compositeSynthetic")):
        fs = found(run(jpeg_xmp(pkt), "a.jpg"))
        assert fs and code in fs[0].summary and "partly" in fs[0].summary


def test_algorithmic_media_is_program_not_ai():
    fs = found(run(jpeg_xmp(xmp(attrs='Iptc4xmpExt:DigitalSourceType="algorithmicMedia"')), "a.jpg"))
    assert fs and "computer program, not by trained AI" in fs[0].summary


@pytest.mark.parametrize("code", ["digitalCapture", "humanEdits", "screenCapture", "composite", "digitalCreation"])
def test_non_ai_iptc_values_not_flagged(code):
    fs = run(jpeg_xmp(xmp(attrs=f'Iptc4xmpExt:DigitalSourceType="http://{DST}{code}"')), "a.jpg")
    assert not found(fs)
    assert code in fs[0].summary


def test_wording_rules():
    fs = run(jpeg_xmp(xmp(attrs='Iptc4xmpExt:DigitalSourceType="trainedAlgorithmicMedia"')), "a.jpg")
    text = " ".join(s for f in fs for s in [f.summary, *f.proves, *f.does_not_prove]).lower()
    for banned in ("authentic", "human-made", "not ai", " true"):
        assert banned not in text
    assert "anyone can add" in text and "not signed" in text


# ---------------------------------------------------------------- containers

def test_png_itxt_compressed_xmp():
    info = PngInfo()
    info.add_itxt("XML:com.adobe.xmp", xmp(attrs='Iptc4xmpExt:DigitalSourceType="trainedAlgorithmicMedia"').decode(), zip=True)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "PNG", pnginfo=info)
    assert found(run(buf.getvalue(), "a.png"))


def test_jpeg_extended_xmp():
    guid = b"0123456789ABCDEF0123456789ABCDEF"
    ext = xmp(f'<Iptc4xmpExt:DigitalSourceType>{"trainedAlgorithmicMedia"}</Iptc4xmpExt:DigitalSourceType>')
    half = len(ext) // 2
    segs = [b"http://ns.adobe.com/xap/1.0/\x00" + xmp()]
    for off, part in ((0, ext[:half]), (half, ext[half:])):
        segs.append(b"http://ns.adobe.com/xmp/extension/\x00" + guid + struct.pack(">II", len(ext), off) + part)
    assert found(run(jpeg(segs), "a.jpg"))


def test_tiff_xmp_tag():
    buf = io.BytesIO()
    img = Image.new("RGB", (8, 8))
    img.save(buf, "TIFF", tiffinfo={700: xmp(attrs='Iptc4xmpExt:DigitalSourceType="trainedAlgorithmicMedia"')})
    assert found(run(buf.getvalue(), "a.tif"))


def test_fallback_scan_in_pdf_and_mp4():
    pkt = b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>' + \
        xmp(attrs='Iptc4xmpExt:DigitalSourceType="trainedAlgorithmicMedia"') + b'<?xpacket end="w"?>'
    pdf = b"%PDF-1.7\n1 0 obj\n<< /Type /Metadata /Subtype /XML >>\nstream\n" + pkt + b"\nendstream\nendobj\n%%EOF"
    assert found(run(pdf, "a.pdf"))
    mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64 + pkt
    assert found(run(mp4, "a.mp4"))


# ---------------------------------------------------------------- safety

def test_doctype_and_entities_are_refused():
    evil = (b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "trainedAlgorithmicMedia">]>'
            + xmp(attrs='Iptc4xmpExt:DigitalSourceType="&a;"'))
    fs = run(jpeg_xmp(evil), "a.jpg")
    assert not found(fs)


def test_billion_laughs_does_not_hang():
    lol = b'<!DOCTYPE lolz [<!ENTITY lol "lol">' + b"".join(
        b'<!ENTITY lol%d "%s">' % (i, (b"&lol%d;" % (i - 1) if i else b"&lol;") * 10) for i in range(1, 10)) + b"]>"
    fs = run(jpeg_xmp(lol + xmp(inner="<x>&lol9;</x>")), "a.jpg")
    assert not found(fs)


def test_oversized_packet_is_ignored():
    assert metadata.parse_xmp(b"<x:xmpmeta>" + b" " * (metadata.MAX_XMP_BYTES + 1) + b"</x:xmpmeta>") is None


def test_garbage_does_not_raise():
    for blob in (b"\xff\xd8\xff\xe1\x00", b"\x89PNG\r\n\x1a\n\x00\x00\xff\xffiTXt", b"II*\x00\xff\xff\xff\xff",
                 b"RIFF\x10\x00\x00\x00WEBPXMP \xff\xff\xff\xff", b"<x:xmpmeta><broken</x:xmpmeta>"):
        fs = run(blob, "a.bin")
        assert fs and not found(fs)


# ---------------------------------------------------------------- AIGC

AIGC = {"Label": "1", "ContentProducer": "PRODUCER001", "ProduceID": "file-9", "ContentPropagator": "", "PropagateID": ""}


@pytest.mark.parametrize("label,word", [("1", "made by AI"), ("2", "may have been made"), ("3", "suspected")])
def test_aigc_png_text_labels(label, word):
    fs = found(run(png_text(AIGC=json.dumps({**AIGC, "Label": label})), "a.png"))
    assert len(fs) == 1 and word in fs[0].summary
    assert fs[0].details["made by (service code)"] == "PRODUCER001"
    assert any("secondary sources" in s for s in fs[0].does_not_prove)


def test_aigc_prefixed_value_in_other_text_key():
    fs = found(run(png_text(Comment="AIGC=" + json.dumps(AIGC)), "a.png"))
    assert fs and fs[0].label.startswith("China AIGC")


def test_aigc_xmp_field_properties():
    ns = 'xmlns:TC260="http://www.tc260.org.cn/ns/AIGC/1.0/"'
    struct_form = xmp("<TC260:AIGC rdf:parseType=\"Resource\"><TC260:Label>1</TC260:Label>"
                      "<TC260:ContentProducer>P1</TC260:ContentProducer></TC260:AIGC>", ns=ns)
    attr_form = xmp(attrs='TC260:Label="1" TC260:ContentProducer="P1"', ns=ns)
    for pkt in (struct_form, attr_form):
        fs = found(run(jpeg_xmp(pkt), "a.jpg"))
        assert fs and "Label 1" in fs[0].summary


def test_aigc_raw_bytes_in_video():
    mp4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 32 + b"AIGC=" + json.dumps(AIGC).encode()
    assert found(run(mp4, "a.mp4"))


def test_aigc_without_known_keys_ignored():
    assert not found(run(png_text(AIGC='{"hello": "world"}'), "a.png"))


# ---------------------------------------------------------------- hints

def test_sd_parameters_needs_real_structure():
    real = "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512"
    assert found(run(png_text(parameters=real), "a.png"))
    assert not found(run(png_text(parameters="exposure +0.3, contrast 10"), "a.png"))


def test_comfyui_prompt_chunk():
    prompt = {"3": {"class_type": "KSampler", "inputs": {}}, "4": {"class_type": "CheckpointLoaderSimple", "inputs": {}}}
    fs = found(run(png_text(prompt=json.dumps(prompt)), "a.png"))
    assert fs and "ComfyUI" in fs[0].summary
    assert not found(run(png_text(prompt="write a poem about the sea"), "a.png"))


def test_software_names_ai_tool():
    fs = found(run(png_text(Software="Midjourney v7"), "a.png"))
    assert fs and "Midjourney" in fs[0].summary and "hint" in fs[0].label
    assert not found(run(png_text(Software="GIMP 2.10"), "a.png"))
    assert not found(run(png_text(Description="Talk about AI, Midjourney and DALL-E"), "a.png"))


def test_exif_software_ai_tool():
    exif = Image.Exif()
    exif[0x0131] = "Adobe Firefly"
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG", exif=exif)
    fs = found(run(buf.getvalue(), "a.jpg"))
    assert fs and "Adobe Firefly" in fs[0].summary


def test_xmp_creator_tool_ai():
    fs = found(run(jpeg_xmp(xmp("<xmp:CreatorTool>DALL-E 3</xmp:CreatorTool>")), "a.jpg"))
    assert fs and "DALL-E" in fs[0].summary
