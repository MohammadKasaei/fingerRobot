#include <linux/build-salt.h>
#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(.gnu.linkonce.this_module) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section(__versions) = {
	{ 0xb3753869, "module_layout" },
	{ 0xe8c708e4, "netdev_info" },
	{ 0x35216b26, "kmalloc_caches" },
	{ 0xbfeda5ab, "register_candev" },
	{ 0xdb710a64, "alloc_can_err_skb" },
	{ 0x16081ffb, "can_dlc2len" },
	{ 0xa715ba7b, "netif_carrier_on" },
	{ 0x26fbb4ae, "netif_carrier_off" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x7ec44287, "__dynamic_netdev_dbg" },
	{ 0xf4a0fd4b, "usb_unanchor_urb" },
	{ 0x4ee39272, "can_bus_off" },
	{ 0xb11289ff, "netif_rx" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0x6d6cd150, "_dev_warn" },
	{ 0x571f5f20, "close_candev" },
	{ 0x7a3d0ceb, "netif_tx_wake_queue" },
	{ 0x3812050a, "_raw_spin_unlock_irqrestore" },
	{ 0x6b7ba003, "usb_deregister" },
	{ 0x90b7a1cf, "alloc_candev_mqs" },
	{ 0x99c37494, "free_candev" },
	{ 0xc141228b, "usb_free_coherent" },
	{ 0x18616700, "_dev_err" },
	{ 0xd5b5fcbe, "can_change_mtu" },
	{ 0x278b8441, "can_change_state" },
	{ 0x3b823a96, "_dev_info" },
	{ 0x43089b53, "usb_submit_urb" },
	{ 0xb601be4c, "__x86_indirect_thunk_rdx" },
	{ 0x141c0839, "unregister_candev" },
	{ 0xbb5bf646, "netif_device_detach" },
	{ 0x962c8ae1, "usb_kill_anchored_urbs" },
	{ 0x6ca86d80, "alloc_can_skb" },
	{ 0xdecd0b29, "__stack_chk_fail" },
	{ 0x4913dc6e, "usb_bulk_msg" },
	{ 0x1d24c881, "___ratelimit" },
	{ 0x8f8fc40f, "kfree_skb" },
	{ 0x2ea2c95c, "__x86_indirect_thunk_rax" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0xa8e112c3, "netdev_err" },
	{ 0xe486b022, "open_candev" },
	{ 0xf5cb25c8, "kmem_cache_alloc_trace" },
	{ 0xc317f50a, "__dynamic_dev_dbg" },
	{ 0x51760917, "_raw_spin_lock_irqsave" },
	{ 0xc2077454, "netdev_warn" },
	{ 0x37a0cba, "kfree" },
	{ 0x69acdf38, "memcpy" },
	{ 0xbf671e22, "usb_register_driver" },
	{ 0xffd1c742, "alloc_canfd_skb" },
	{ 0x8762619a, "can_len2dlc" },
	{ 0x29361773, "complete" },
	{ 0xa586aa90, "can_get_echo_skb" },
	{ 0x32d99113, "consume_skb" },
	{ 0xf98013c3, "usb_alloc_coherent" },
	{ 0x3e2a105a, "can_put_echo_skb" },
	{ 0x45221b5b, "devm_kmalloc" },
	{ 0x4d1ff60a, "wait_for_completion_timeout" },
	{ 0x9e28038d, "can_free_echo_skb" },
	{ 0xc9d8c16a, "usb_free_urb" },
	{ 0x3b63bdd1, "usb_anchor_urb" },
	{ 0x51bd55b5, "completion_done" },
	{ 0x34ac35f4, "usb_alloc_urb" },
	{ 0xe914e41e, "strcpy" },
};

MODULE_INFO(depends, "can-dev");

MODULE_ALIAS("usb:v0BFDp000Ad*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp000Bd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp000Cd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp000Ed*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp000Fd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0010d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0011d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0012d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0013d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0016d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0017d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0018d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0019d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp001Ad*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp001Bd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp001Cd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp001Dd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0022d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0023d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0027d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0120d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0121d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0122d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0123d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0124d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0126d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0127d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0128d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0004d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0002d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0005d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0003d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0102d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0104d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0105d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0106d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0107d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0108d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0109d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp010Ad*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp010Bd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp010Cd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp010Dd*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp010Ed*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0111d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0112d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0113d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0114d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0115d*dc*dsc*dp*ic*isc*ip*in*");
MODULE_ALIAS("usb:v0BFDp0116d*dc*dsc*dp*ic*isc*ip*in*");

MODULE_INFO(srcversion, "576669880C2507ECDEF86BF");
