#!/usr/bin/env python3
"""
Publication-grade packet structure figures for ARP and IPv4/TCP
RFC 826, RFC 791, RFC 9293

Output: high-resolution PNG (easily switchable to PDF/SVG)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Helpers ----------

def draw_box(ax, x, y, w, h, label, fs=8):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.2))
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center", fontsize=fs, wrap=True)

def section_title(ax, y, text):
    ax.text(50, y, text, ha="center", va="center",
            fontsize=12, fontweight="bold")

# ---------- ARP Figure ----------

def generate_arp_figure(path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 95, "ARP Packet Format (RFC 826)",
            ha="center", va="center", fontsize=16, fontweight="bold")

    # Ethernet header
    section_title(ax, 88, "Ethernet Frame")
    draw_box(ax, 5, 78, 30, 8, "Destination MAC\n(6 bytes)")
    draw_box(ax, 35, 78, 30, 8, "Source MAC\n(6 bytes)")
    draw_box(ax, 65, 78, 30, 8, "EtherType\n0x0806 (ARP)")

    # ARP payload
    section_title(ax, 70, "ARP Payload (28 bytes)")

    y = 60
    h = 8

    draw_box(ax, 5, y, 45, h, "Hardware Type (HTYPE)\nEthernet = 1")
    draw_box(ax, 50, y, 45, h, "Protocol Type (PTYPE)\nIPv4 = 0x0800")

    y -= h
    draw_box(ax, 5, y, 20, h, "HLEN\n6")
    draw_box(ax, 25, y, 20, h, "PLEN\n4")
    draw_box(ax, 45, y, 50, h, "Opcode\n1 = Request\n2 = Reply")

    y -= h
    draw_box(ax, 5, y, 90, h, "Sender Hardware Address (SHA)\nMAC Address")

    y -= h
    draw_box(ax, 5, y, 90, h, "Sender Protocol Address (SPA)\nIPv4 Address")

    y -= h
    draw_box(ax, 5, y, 90, h,
             "Target Hardware Address (THA)\nMAC Address (zeroed in requests)")

    y -= h
    draw_box(ax, 5, y, 90, h, "Target Protocol Address (TPA)\nIPv4 Address")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- IPv4 + TCP Figure ----------

def generate_ipv4_tcp_figure(path):
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.axis("off")

    ax.text(50, 124,
            "IPv4 and TCP Packet Structure (RFC 791 / RFC 9293)",
            ha="center", va="center", fontsize=16, fontweight="bold")

    # Ethernet
    section_title(ax, 116, "Ethernet Frame")
    draw_box(ax, 5, 106, 30, 8, "Destination MAC")
    draw_box(ax, 35, 106, 30, 8, "Source MAC")
    draw_box(ax, 65, 106, 30, 8, "EtherType\n0x0800 (IPv4)")

    # IPv4 Header
    section_title(ax, 98, "IPv4 Header (minimum 20 bytes)")
    y = 88
    h = 7

    draw_box(ax, 5, y, 20, h, "Version\n4")
    draw_box(ax, 25, y, 20, h, "IHL")
    draw_box(ax, 45, y, 25, h, "DSCP / ECN")
    draw_box(ax, 70, y, 25, h, "Total Length")

    y -= h
    draw_box(ax, 5, y, 45, h, "Identification")
    draw_box(ax, 50, y, 20, h, "Flags")
    draw_box(ax, 70, y, 25, h, "Fragment Offset")

    y -= h
    draw_box(ax, 5, y, 30, h, "TTL")
    draw_box(ax, 35, y, 30, h, "Protocol\nTCP = 6")
    draw_box(ax, 65, y, 30, h, "Header Checksum")

    y -= h
    draw_box(ax, 5, y, 90, h, "Source IPv4 Address")

    y -= h
    draw_box(ax, 5, y, 90, h, "Destination IPv4 Address")

    y -= h
    draw_box(ax, 5, y, 90, h, "Options (optional)")

    # TCP Header
    section_title(ax, y - 6, "TCP Header (minimum 20 bytes)")
    y -= 16

    draw_box(ax, 5, y, 45, h, "Source Port")
    draw_box(ax, 50, y, 45, h, "Destination Port")

    y -= h
    draw_box(ax, 5, y, 90, h, "Sequence Number")

    y -= h
    draw_box(ax, 5, y, 90, h, "Acknowledgment Number")

    y -= h
    draw_box(ax, 5, y, 20, h, "Data Offset")
    draw_box(ax, 25, y, 20, h, "Reserved")
    draw_box(ax, 45, y, 50, h,
             "Flags\nSYN ACK FIN RST PSH URG ECE CWR")

    y -= h
    draw_box(ax, 5, y, 45, h, "Window Size")
    draw_box(ax, 50, y, 45, h, "Checksum")

    y -= h
    draw_box(ax, 5, y, 90, h, "Urgent Pointer / Options")

    y -= h
    draw_box(ax, 5, y, 90, h, "TCP Payload (Data)")

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- Main ----------

if __name__ == "__main__":
    generate_arp_figure("arp_packet_structure.svg")
    generate_ipv4_tcp_figure("ipv4_tcp_packet_structure.svg")
